"""Hand and action abstraction utilities for Texas Hold'em.

This module provides both a legacy rank-based hand bucketing (retained for
backward compatibility) and a modern Expected Hand Strength (EHS) bucketing
abstraction used by world-class poker AI systems (Libratus, Pluribus,
DeepStack).  EHS estimates the probability of winning at showdown against a
random opponent hand, sampled by Monte Carlo simulation using the ``treys``
poker hand evaluator.

Key features
------------
* :func:`expected_hand_strength` - Monte Carlo EHS estimation for any
  hole-card / board-card combination.
* :func:`ehs_bucket` - Map an EHS value in ``[0, 1]`` to a discrete bucket
  index suitable for information-set abstraction in CFR variants.
* :func:`preflop_hand_class` - Return the canonical 169-class preflop hand
  index (e.g. ``AAo``, ``AKs``, ``72o``) for cache lookup.
* :func:`bucket_hand_ehs` - EHS-based bucketing with on-disk caching of the
  1326 preflop hand EHS values for fast repeated lookup.
* :func:`bucket_hand` - Legacy rank-based bucketing (2 buckets) retained for
  backward compatibility with existing callers and tests.
* :func:`abstract_action` - Map concrete actions to aggressive/passive
  buckets for action abstraction.

The EHS cache is stored at ``~/.cache/poker_ai/ehs_preflop_*.npy`` and is
populated lazily on first use.  A small number of Monte Carlo simulations
(default 2000 per hand) yields EHS estimates with standard error < 0.01,
which is well below the resolution of typical bucket counts (10-200).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import TypedDict

import numpy as np

# ``treys`` is a hard dependency for EHS computation.  When it is unavailable
# we fall back to the legacy rank-based abstraction so that imports never
# fail in environments that only need the simple bucketing.
try:  # pragma: no cover - exercised implicitly when treys is installed
    from treys import Card, Evaluator  # type: ignore

    _TREYS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    Card = None  # type: ignore[assignment]
    Evaluator = None  # type: ignore[assignment]
    _TREYS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Legacy rank-based bucketing (backward compatibility)
# ---------------------------------------------------------------------------


class HandBucket(TypedDict):
    """Mapping from a set of rank labels to a bucket index."""

    ranks: list[str]
    bucket: int


HAND_BUCKETS: list[HandBucket] = [
    {"ranks": ["A", "K"], "bucket": 0},
    {"ranks": ["Q", "J"], "bucket": 1},
]


def bucket_hand(hand: tuple[str, str]) -> int:
    """Return a bucket index for the given hand based on high card.

    .. deprecated::
        This legacy implementation only distinguishes two buckets.  Prefer
        :func:`bucket_hand_ehs` for any new code.
    """
    ranks = [card[0] for card in hand]
    for mapping in HAND_BUCKETS:
        if any(r in ranks for r in mapping["ranks"]):
            return mapping["bucket"]
    return len(HAND_BUCKETS)


def abstract_action(action: str) -> str:
    """Map concrete actions to abstracted buckets."""
    action = action.lower()
    if action in {"bet", "raise"}:
        return "aggressive"
    if action in {"check", "call"}:
        return "passive"
    return action


# ---------------------------------------------------------------------------
# Expected Hand Strength (EHS) bucketing
# ---------------------------------------------------------------------------

# Canonical rank order used throughout the module.
RANKS_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_INDEX = {r: i for i, r in enumerate(RANKS_ORDER)}
SUITS = ["s", "h", "d", "c"]
_FULL_DECK = [r + s for r in RANKS_ORDER for s in SUITS]

# Singleton evaluator reused across all EHS computations.  ``treys.Evaluator``
# is stateless (no learned weights) so a single instance is safe to share.
_EVALUATOR: "Evaluator | None" = None  # type: ignore[type-arg]


def _get_evaluator() -> "Evaluator":  # type: ignore[type-arg]
    global _EVALUATOR
    if _EVALUATOR is None:
        if not _TREYS_AVAILABLE:
            raise RuntimeError(
                "treys is required for EHS computation; install with `pip install treys`"
            )
        _EVALUATOR = Evaluator()
    return _EVALUATOR


def preflop_hand_class(hole: tuple[str, str] | list[str]) -> str:
    """Return the canonical 169-class preflop hand label.

    The label is formed by ordering the two ranks descending and appending
    ``s`` (suited), ``o`` (offsuit), or nothing for pairs.  Examples::

        ("Ah", "Ad") -> "AA"
        ("Ah", "Kh") -> "AKs"
        ("Ah", "Kd") -> "AKo"

    This gives 13 pairs + 78 suited + 78 offsuit = 169 classes, which is the
    canonical preflop abstraction used in poker literature.
    """
    if len(hole) != 2:
        raise ValueError(f"Expected exactly 2 hole cards, got {len(hole)}")
    c0, c1 = hole[0], hole[1]
    r0, r1 = c0[0], c1[0]
    s0, s1 = c0[1], c1[1]
    # Order ranks descending.  Handle the wheel (A-5) by keeping Ace high
    # at preflop - the wheel only matters for post-flop straight detection.
    if RANK_INDEX[r0] < RANK_INDEX[r1]:
        r0, r1 = r1, r0
    if r0 == r1:
        return f"{r0}{r1}"
    suffix = "s" if s0 == s1 else "o"
    return f"{r0}{r1}{suffix}"


def expected_hand_strength(
    hole: tuple[str, str] | list[str],
    board: tuple[str, ...] | list[str] | None = None,
    n_simulations: int = 2000,
    rng: random.Random | None = None,
) -> float:
    """Estimate the probability of winning at showdown against a random opponent.

    Uses Monte Carlo simulation: for each iteration, complete the board (if
    needed) and deal a random opponent hole card, then evaluate both hands.
    Ties count as half a win.  Returns a float in ``[0, 1]``.

    Parameters
    ----------
    hole:
        Two hole cards (e.g. ``["Ah", "Kd"]``).
    board:
        Zero to five community cards.  If fewer than five, the remaining
        cards are sampled randomly from the depleted deck.
    n_simulations:
        Number of Monte Carlo trials.  Default 2000 gives standard error
        < 0.01 for win probabilities near 0.5.
    rng:
        Optional ``random.Random`` instance for reproducibility.
    """
    if not _TREYS_AVAILABLE:
        raise RuntimeError(
            "treys is required for EHS computation; install with `pip install treys`"
        )
    if len(hole) != 2:
        raise ValueError(f"Expected exactly 2 hole cards, got {len(hole)}")
    board = list(board) if board else []
    if len(board) > 5:
        raise ValueError(f"Board has {len(board)} cards; maximum is 5")
    rng = rng or random.Random()
    evaluator = _get_evaluator()
    hole_cards = [Card.new(c) for c in hole]
    board_cards = [Card.new(c) for c in board]
    known = set(hole) | set(board)
    remaining_deck = [c for c in _FULL_DECK if c not in known]
    if len(remaining_deck) < 2 + (5 - len(board)):
        raise ValueError("Not enough cards remaining in deck for simulation")

    wins = 0.0
    for _ in range(n_simulations):
        deck = remaining_deck[:]
        rng.shuffle(deck)
        needed_board = 5 - len(board)
        sim_board = board_cards + [Card.new(c) for c in deck[:needed_board]]
        deck = deck[needed_board:]
        opp_hole = [Card.new(c) for c in deck[:2]]
        my_score = evaluator.evaluate(sim_board, hole_cards)
        opp_score = evaluator.evaluate(sim_board, opp_hole)
        # treys: lower is better.
        if my_score < opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5
    return wins / n_simulations


def ehs_bucket(ehs: float, num_buckets: int = 10) -> int:
    """Map an EHS value in ``[0, 1]`` to a discrete bucket index.

    Uses equal-width bucketing (each bucket spans ``1/num_buckets`` of the
    EHS range).  Bucket 0 is the weakest hand, bucket ``num_buckets-1`` is
    the strongest.  Values outside ``[0, 1]`` are clamped.
    """
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive, got {num_buckets}")
    ehs_clamped = max(0.0, min(1.0, float(ehs)))
    bucket = int(ehs_clamped * num_buckets)
    # Guard against ehs == 1.0 mapping to bucket == num_buckets.
    return min(bucket, num_buckets - 1)


# ---------------------------------------------------------------------------
# Preflop EHS cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(
    os.environ.get(
        "POKER_AI_CACHE_DIR", str(Path.home() / ".cache" / "poker_ai")
    )
)


def _preflop_cache_path(n_simulations: int) -> Path:
    return _CACHE_DIR / f"ehs_preflop_{n_simulations}.npy"


def _ensure_preflop_cache(n_simulations: int, rng_seed: int = 42) -> np.ndarray:
    """Load or build the 1326-entry preflop EHS cache.

    The cache is a length-1326 numpy array indexed by the lexicographic
    order of ``itertools.combinations(deck, 2)``.  It is stored as a
    ``.npy`` file in ``~/.cache/poker_ai/`` (or ``$POKER_AI_CACHE_DIR``).
    """
    cache_path = _preflop_cache_path(n_simulations)
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            # Corrupt cache file - rebuild.
            pass

    from itertools import combinations

    hands = list(combinations(_FULL_DECK, 2))
    cache = np.zeros(len(hands), dtype=np.float32)
    rng = random.Random(rng_seed)
    for idx, hole in enumerate(hands):
        cache[idx] = expected_hand_strength(
            list(hole), board=None, n_simulations=n_simulations, rng=rng
        )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # ``np.save`` appends ``.npy`` to the path it is given, so we pass a
    # path *without* the ``.npy`` suffix and then rename the resulting
    # file.  This gives an atomic rename from ``ehs_preflop_N.tmp.npy``
    # to ``ehs_preflop_N.npy`` (both filenames include the ``.npy`` suffix).
    tmp_path_no_suffix = cache_path.with_suffix(".tmp")
    np.save(tmp_path_no_suffix, cache)
    # ``np.save`` actually wrote to ``tmp_path_no_suffix`` + ".npy".
    actual_tmp_path = tmp_path_no_suffix.with_suffix(".tmp.npy")
    actual_tmp_path.replace(cache_path)
    return cache


def bucket_hand_ehs(
    hole: tuple[str, str] | list[str],
    board: tuple[str, ...] | list[str] | None = None,
    num_buckets: int = 10,
    n_simulations: int = 2000,
    use_cache: bool = True,
) -> int:
    """Bucket a hand using Expected Hand Strength.

    For preflop (empty board), a cached 1326-entry EHS table is used for
    O(1) lookup when ``use_cache=True``.  For postflop, EHS is computed
    on-the-fly via Monte Carlo simulation.

    Parameters
    ----------
    hole:
        Two hole cards.
    board:
        Zero to five community cards.
    num_buckets:
        Number of EHS buckets to map into.  10 is a common default; top
        poker AI systems use 50-200 buckets for fine-grained abstraction.
    n_simulations:
        Monte Carlo sample count for postflop EHS (preflop uses the cached
        table which is typically computed with 2000-10000 simulations).
    use_cache:
        Whether to use the preflop cache.  When ``False``, always compute
        EHS on-the-fly (useful for testing or when the cache is stale).
    """
    if not _TREYS_AVAILABLE:
        # Graceful fallback to legacy bucketing.
        return bucket_hand(tuple(hole))

    board = list(board) if board else []
    if not board and use_cache:
        # Preflop: use cached table.
        from itertools import combinations

        cache = _ensure_preflop_cache(n_simulations)
        # Find the index of this hole in the combinations order.
        hole_sorted = tuple(sorted(hole))
        hands = list(combinations(_FULL_DECK, 2))
        try:
            idx = hands.index(hole_sorted)  # type: ignore[arg-type]
        except ValueError:
            # Should not happen for valid hole cards.
            return 0
        ehs = float(cache[idx])
    else:
        ehs = expected_hand_strength(
            hole, board=board, n_simulations=n_simulations
        )
    return ehs_bucket(ehs, num_buckets=num_buckets)


__all__ = [
    "HandBucket",
    "HAND_BUCKETS",
    "bucket_hand",
    "abstract_action",
    "preflop_hand_class",
    "expected_hand_strength",
    "ehs_bucket",
    "bucket_hand_ehs",
]
