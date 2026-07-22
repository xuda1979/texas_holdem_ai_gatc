"""Tests for poker_ai.utils.abstraction.

Covers both the legacy rank-based bucketing and the new EHS-based
(Expected Hand Strength) bucketing used by world-class poker AI systems.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.utils.abstraction import (  # noqa: E402
    HAND_BUCKETS,
    abstract_action,
    bucket_hand,
    bucket_hand_ehs,
    ehs_bucket,
    expected_hand_strength,
    preflop_hand_class,
)

try:
    import treys  # noqa: F401

    TREYS_AVAILABLE = True
except Exception:
    TREYS_AVAILABLE = False

treys_required = pytest.mark.skipif(
    not TREYS_AVAILABLE, reason="treys not installed"
)


# ---------------------------------------------------------------------------
# Legacy rank-based bucketing
# ---------------------------------------------------------------------------


def test_legacy_bucket_hand_high_cards() -> None:
    """Hands containing A or K map to bucket 0."""
    assert bucket_hand(("Ah", "Kd")) == 0
    assert bucket_hand(("As", "2c")) == 0
    assert bucket_hand(("Kh", "2d")) == 0


def test_legacy_bucket_hand_queen_jack() -> None:
    """Hands containing Q or J (but not A/K) map to bucket 1."""
    assert bucket_hand(("Qh", "Jd")) == 1
    assert bucket_hand(("Qs", "2c")) == 1
    assert bucket_hand(("Jh", "2d")) == 1


def test_legacy_bucket_hand_low_cards() -> None:
    """Hands without A/K/Q/J map to the fallback bucket."""
    assert bucket_hand(("2c", "3d")) == len(HAND_BUCKETS)
    assert bucket_hand(("Tc", "9d")) == len(HAND_BUCKETS)


def test_abstract_action_categories() -> None:
    """Aggressive / passive / other action mapping."""
    assert abstract_action("bet") == "aggressive"
    assert abstract_action("raise") == "aggressive"
    assert abstract_action("RAISE") == "aggressive"  # case-insensitive
    assert abstract_action("check") == "passive"
    assert abstract_action("call") == "passive"
    assert abstract_action("fold") == "fold"  # unchanged
    # Unknown actions are lowercased and passed through unchanged.
    assert abstract_action("All-In") == "all-in"


# ---------------------------------------------------------------------------
# Preflop hand class labels (169 canonical classes)
# ---------------------------------------------------------------------------


def test_preflop_hand_class_pairs() -> None:
    """Pairs have no suited/offsuit suffix."""
    assert preflop_hand_class(["Ah", "Ad"]) == "AA"
    assert preflop_hand_class(["Kh", "Kd"]) == "KK"
    assert preflop_hand_class(["2c", "2d"]) == "22"


def test_preflop_hand_class_suited() -> None:
    """Suited hands end with 's'."""
    assert preflop_hand_class(["Ah", "Kh"]) == "AKs"
    assert preflop_hand_class(["Kh", "Ah"]) == "AKs"  # order-independent
    assert preflop_hand_class(["2c", "3c"]) == "32s"  # 3 high


def test_preflop_hand_class_offsuit() -> None:
    """Offsuit hands end with 'o'."""
    assert preflop_hand_class(["Ah", "Kd"]) == "AKo"
    assert preflop_hand_class(["Kh", "Ad"]) == "AKo"  # order-independent
    assert preflop_hand_class(["2c", "3d"]) == "32o"


def test_preflop_hand_class_invalid() -> None:
    """Invalid hole card counts raise ValueError."""
    with pytest.raises(ValueError):
        preflop_hand_class(["Ah"])
    with pytest.raises(ValueError):
        preflop_hand_class(["Ah", "Kd", "Qc"])


# ---------------------------------------------------------------------------
# EHS bucketing helper
# ---------------------------------------------------------------------------


def test_ehs_bucket_boundaries() -> None:
    """EHS values map to the correct bucket index."""
    assert ehs_bucket(0.0, 10) == 0
    assert ehs_bucket(0.099, 10) == 0
    assert ehs_bucket(0.1, 10) == 1
    assert ehs_bucket(0.5, 10) == 5
    assert ehs_bucket(0.95, 10) == 9
    assert ehs_bucket(1.0, 10) == 9  # clamped


def test_ehs_bucket_clamping() -> None:
    """Out-of-range EHS values are clamped."""
    assert ehs_bucket(-0.5, 10) == 0
    assert ehs_bucket(1.5, 10) == 9


def test_ehs_bucket_invalid_count() -> None:
    """Non-positive bucket counts raise ValueError."""
    with pytest.raises(ValueError):
        ehs_bucket(0.5, 0)
    with pytest.raises(ValueError):
        ehs_bucket(0.5, -1)


# ---------------------------------------------------------------------------
# Monte Carlo EHS estimation (requires treys)
# ---------------------------------------------------------------------------


@treys_required
def test_expected_hand_strength_aa_beats_23o() -> None:
    """AA has substantially higher EHS than 23o."""
    rng = random.Random(42)
    ehs_aa = expected_hand_strength(["Ah", "Ad"], n_simulations=300, rng=rng)
    rng2 = random.Random(42)
    ehs_23 = expected_hand_strength(["2c", "3d"], n_simulations=300, rng=rng2)
    assert 0.7 < ehs_aa < 0.95, f"AA EHS out of range: {ehs_aa}"
    assert 0.2 < ehs_23 < 0.45, f"23o EHS out of range: {ehs_23}"
    assert ehs_aa > ehs_23


@treys_required
def test_expected_hand_strength_with_flop() -> None:
    """EHS computation works with a partial board."""
    rng = random.Random(42)
    # AhKh on a QhJh2d flop - strong made hand + royal flush draw.
    ehs = expected_hand_strength(
        ["Ah", "Kh"], board=["Qh", "Jh", "2d"], n_simulations=300, rng=rng
    )
    assert ehs > 0.6, f"AKs on QJ2hh should have EHS > 0.6, got {ehs}"


@treys_required
def test_expected_hand_strength_invalid_inputs() -> None:
    """Invalid hand/board combinations raise ValueError."""
    with pytest.raises(ValueError):
        expected_hand_strength(["Ah"], n_simulations=10)
    with pytest.raises(ValueError):
        expected_hand_strength(["Ah", "Kd", "Qc"], n_simulations=10)
    with pytest.raises(ValueError):
        expected_hand_strength(
            ["Ah", "Kd"], board=["2c", "3d", "4c", "5s", "6h", "7c"], n_simulations=10
        )


@treys_required
def test_expected_hand_strength_reproducible() -> None:
    """Same RNG seed produces identical EHS estimates."""
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    ehs1 = expected_hand_strength(["Ah", "Ad"], n_simulations=200, rng=rng1)
    ehs2 = expected_hand_strength(["Ah", "Ad"], n_simulations=200, rng=rng2)
    assert ehs1 == ehs2


# ---------------------------------------------------------------------------
# EHS-based hand bucketing
# ---------------------------------------------------------------------------


@treys_required
def test_bucket_hand_ehs_ordering() -> None:
    """Stronger hands map to higher (better) buckets."""
    b_aa = bucket_hand_ehs(
        ["Ah", "Ad"], num_buckets=10, n_simulations=200, use_cache=False
    )
    b_23 = bucket_hand_ehs(
        ["2c", "3d"], num_buckets=10, n_simulations=200, use_cache=False
    )
    assert b_aa > b_23, f"AA ({b_aa}) should bucket higher than 23o ({b_23})"


@treys_required
def test_bucket_hand_ehs_with_board() -> None:
    """EHS bucketing works with a partial board."""
    b_made_hand = bucket_hand_ehs(
        ["Ah", "Ad"], board=["As", "Kd", "2c"], num_buckets=10, n_simulations=200
    )
    b_trash = bucket_hand_ehs(
        ["2c", "3d"], board=["As", "Kd", "Qc"], num_buckets=10, n_simulations=200
    )
    assert b_made_hand > b_trash


@treys_required
def test_bucket_hand_ehs_returns_in_range() -> None:
    """Bucket index is always in [0, num_buckets-1]."""
    for hole in [("Ah", "Ad"), ("2c", "3d"), ("Kh", "Qh"), ("7s", "2d")]:
        b = bucket_hand_ehs(
            list(hole), num_buckets=5, n_simulations=100, use_cache=False
        )
        assert 0 <= b < 5, f"Hole {hole} got bucket {b} out of [0, 5)"


def test_bucket_hand_ehs_fallback_without_treys(monkeypatch: pytest.MonkeyPatch) -> None:
    """When treys is unavailable, EHS bucketing falls back to legacy bucketing."""
    from poker_ai.utils import abstraction

    if abstraction._TREYS_AVAILABLE:
        monkeypatch.setattr(abstraction, "_TREYS_AVAILABLE", False)
        b = abstraction.bucket_hand_ehs(
            ["Ah", "Kd"], num_buckets=10, n_simulations=10, use_cache=False
        )
        # Legacy bucket_hand returns 0 for hands with A or K.
        assert b == 0
    else:  # pragma: no cover - exercised only when treys is genuinely absent
        b = abstraction.bucket_hand_ehs(
            ["Ah", "Kd"], num_buckets=10, n_simulations=10, use_cache=False
        )
        assert b == 0


@treys_required
def test_preflop_ehs_cache_build_and_lookup() -> None:
    """Preflop cache builds and returns consistent values across calls."""
    from poker_ai.utils.abstraction import _ensure_preflop_cache

    # Use a tiny simulation count to keep the test fast.
    cache = _ensure_preflop_cache(50)
    assert cache.shape == (1326,)
    assert (cache >= 0).all() and (cache <= 1).all()
    # AA should be near the top.
    assert cache.max() > 0.7
    # Re-call should return the same array (from cache).
    cache2 = _ensure_preflop_cache(50)
    assert cache is cache2 or (cache == cache2).all()
