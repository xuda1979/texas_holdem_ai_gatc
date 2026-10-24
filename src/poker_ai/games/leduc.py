"""Leduc poker — a standard benchmark between Kuhn and full Texas Hold'em.

Leduc hold'em (Southey et al., 2005) is a two-player, two-round, zero-sum
limit poker game used widely to benchmark CFR and related equilibrium
solvers:

- **Deck**: 6 cards — ranks J, Q, K, each in two suits.  A single
  public card is revealed after the first betting round.
- **Betting**: Each round has a limit of two raises.  Preflop raises
  add ``2`` chips; postflop raises add ``4`` chips (the standard
  Southey et al. betting structure; the equilibrium value of the game
  is ≈ −0.0856 for the first actor).
- **Showdown**: If the private hole card pairs the public card that
  player wins; otherwise the higher rank wins; equal ranks (same rank,
  different suits) split the pot.

This module provides:

- :class:`LeducGame` — pure-Python game-tree implementation.
- :func:`leduc_cfr` — vanilla CFR trainer returning the average strategy.
- :func:`best_response_value` — exact best-response value for a player.
- :func:`exploitability` — exact exploitability ``(br0 + br1) / 2``,
  the standard Nash-convergence metric (Zinkevich et al., 2007).

The game tree is small enough (~556 information sets) that exact
best-response is tractable, making this the first real exploitability
benchmark in the repo beyond the matrix-game helper in
``evaluation/exploitability.py`` and the tiny Kuhn solver.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

RANKS: Tuple[str, ...] = ("J", "Q", "K")
SUITS: Tuple[str, ...] = ("s", "h")
DECK: Tuple[str, ...] = tuple(r + s for r in RANKS for s in SUITS)
RANK_VALUE: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}

# Actions: fold, call, raise.
ACTIONS: Tuple[str, ...] = ("f", "c", "r")
ACTION_INDEX: Dict[str, int] = {"f": 0, "c": 1, "r": 2}

PREFLOP_RAISE = 2
POSTFLOP_RAISE = 4
MAX_RAISES_PER_ROUND = 2
ANTE = 1


def _rank(card: str) -> int:
    return RANK_VALUE[card[0]]


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


@dataclass
class LeducState:
    """Mutable Leduc game state.

    The ``dealt_public`` field carries the public card that *will* be
    revealed after the preflop round; it is set at deal time so CFR can
    enumerate all chance outcomes.  The ``public`` field is ``None``
    preflop and set to ``dealt_public`` once the flop is revealed,
    ensuring the preflop information set does not leak the public card.
    """

    hole: Tuple[str, str]
    dealt_public: str
    public: Optional[str] = None
    history: str = ""
    round: int = 0
    pot: int = 2 * ANTE
    contributions: Tuple[int, int] = (ANTE, ANTE)
    raises_this_round: int = 0
    acted_this_round: Tuple[bool, bool] = (False, False)
    current_player: int = 0
    terminal: bool = False
    folded: Tuple[bool, bool] = (False, False)

    def info_set(self, player: int) -> str:
        """Information-set key for ``player`` at this state.

        Preflop the key is ``"<hole><round><history>"``; postflop the
        public card is included: ``"<hole><round><public><history>"``.
        """
        pub = self.public if (self.public is not None and self.round == 1) else ""
        return f"{self.hole[player]}{self.round}{pub}{self.history}"


class LeducGame:
    """Pure-Python Leduc hold'em game-tree implementation."""

    # ------------------------------------------------------------------
    # Deal
    # ------------------------------------------------------------------
    def deal(
        self,
        hole_cards: Optional[Tuple[str, str]] = None,
        public_card: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> LeducState:
        rng = rng or random
        if hole_cards is not None:
            hole = hole_cards
        else:
            hole = tuple(rng.sample(list(DECK), 2))  # type: ignore[arg-type]
        if public_card is not None:
            public = public_card
        else:
            remaining = [c for c in DECK if c not in hole]
            public = rng.choice(remaining)
        return LeducState(
            hole=hole,  # type: ignore[arg-type]
            dealt_public=public,
            public=None,
            history="",
            round=0,
            pot=2 * ANTE,
            contributions=(ANTE, ANTE),
            raises_this_round=0,
            acted_this_round=(False, False),
            current_player=0,
            terminal=False,
            folded=(False, False),
        )

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------
    def legal_actions(self, state: LeducState) -> List[str]:
        if state.terminal:
            return []
        if state.raises_this_round >= MAX_RAISES_PER_ROUND:
            return ["f", "c"]
        return ["f", "c", "r"]

    def apply_action(self, state: LeducState, action: str) -> LeducState:
        if action not in self.legal_actions(state):
            raise ValueError(
                f"illegal action {action!r} at history {state.history!r}"
            )
        player = state.current_player
        other = 1 - player
        contribs = list(state.contributions)
        acted = list(state.acted_this_round)
        folded = list(state.folded)
        history = state.history + action
        raises = state.raises_this_round
        pot = state.pot
        round_ = state.round
        public = state.public
        terminal = False
        next_player = other

        acted[player] = True

        if action == "f":
            folded[player] = True
            terminal = True
        elif action == "c":
            diff = contribs[other] - contribs[player]
            contribs[player] += diff
            pot += diff
            # Check if the betting round closes.
            if acted[other] and contribs[0] == contribs[1]:
                round_, public, raises, acted, next_player = self._close_round(
                    state
                )
        elif action == "r":
            raise_size = PREFLOP_RAISE if state.round == 0 else POSTFLOP_RAISE
            diff = contribs[other] - contribs[player]
            contribs[player] += diff + raise_size
            pot += diff + raise_size
            raises += 1
            # A raise re-opens the betting round: the opponent must respond,
            # so clear their acted flag.  If the raise cap is already hit
            # the opponent can only call/fold, but they still must act.
            acted[other] = False

        return LeducState(
            hole=state.hole,
            dealt_public=state.dealt_public,
            public=public,
            history=history,
            round=round_,
            pot=pot,
            contributions=(contribs[0], contribs[1]),
            raises_this_round=raises,
            acted_this_round=(acted[0], acted[1]),
            current_player=next_player,
            terminal=terminal,
            folded=(folded[0], folded[1]),
        )

    def _close_round(
        self, state: LeducState
    ) -> Tuple[int, Optional[str], int, Tuple[bool, bool], int]:
        """Close the current betting round and open the next (or showdown).

        Returns ``(round, public, raises_this_round, acted, next_player)``.
        """
        if state.round == 0:
            # Reveal the public card; the same player (0) acts first
            # postflop — standard Leduc keeps a fixed first actor across
            # both rounds (Southey et al., 2005).
            return 1, state.dealt_public, 0, (False, False), 0
        # Postflop round closed -> showdown (terminal handled by is_terminal).
        return 1, state.public, 0, (True, True), 0

    def is_terminal(self, state: LeducState) -> bool:
        if state.terminal:
            return True
        # Showdown: postflop round closed with both players having acted
        # and matched contributions, no fold.
        if (
            state.round == 1
            and state.public is not None
            and state.acted_this_round == (True, True)
            and state.contributions[0] == state.contributions[1]
            and not any(state.folded)
        ):
            return True
        return False

    def terminal_utility(self, state: LeducState, player: int) -> float:
        """Net chip utility for ``player`` at a terminal state.

        Winner gains the opponent's total contribution; loser loses
        their own contribution.  At a fold the non-folding player wins
        the pot.
        """
        if not self.is_terminal(state):
            raise ValueError("state is not terminal")
        if any(state.folded):
            # The non-folding player wins the pot.
            winner = 1 if state.folded[0] else 0
            if player == winner:
                return float(state.contributions[1 - player])
            return -float(state.contributions[player])
        # Showdown.
        hole = state.hole
        public = state.public
        assert public is not None, "public must be revealed at showdown"
        p0_pair = hole[0][0] == public[0]
        p1_pair = hole[1][0] == public[0]
        if p0_pair and not p1_pair:
            winner = 0
        elif p1_pair and not p0_pair:
            winner = 1
        elif _rank(hole[0]) > _rank(hole[1]):
            winner = 0
        elif _rank(hole[1]) > _rank(hole[0]):
            winner = 1
        else:
            # Tie (same rank, different suits): split the pot.  With two
            # suits per rank a both-pair showdown is impossible, so ties
            # are exactly the equal-rank-no-pair deals (1 in 5).
            return 0.0
        if player == winner:
            return float(state.contributions[1 - player])
        return -float(state.contributions[player])


# ---------------------------------------------------------------------------
# CFR
# ---------------------------------------------------------------------------


@dataclass
class LeducNode:
    """Information-set node for Leduc CFR (3 actions: f/c/r)."""

    info_set: str
    regret_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    strategy_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def get_strategy(
        self, realization_weight: float, legal: Optional[List[str]] = None
    ) -> List[float]:
        """Regret-matched strategy restricted to ``legal`` actions.

        Returns a full ``ACTIONS``-length vector with zero probability
        on illegal actions, so callers can index by ``ACTION_INDEX``.
        """
        if legal is None:
            legal = list(ACTIONS)
        positive = [max(self.regret_sum[ACTION_INDEX[a]], 0.0) for a in legal]
        total = sum(positive)
        if total > 0:
            strat = [p / total for p in positive]
        else:
            strat = [1.0 / len(legal)] * len(legal)
        for a, s in zip(legal, strat):
            self.strategy_sum[ACTION_INDEX[a]] += realization_weight * s
        strategy = [0.0] * len(ACTIONS)
        for a, s in zip(legal, strat):
            strategy[ACTION_INDEX[a]] = s
        return strategy

    def get_average_strategy(self) -> List[float]:
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / 3.0] * 3


def leduc_cfr(
    iterations: int, seed: Optional[int] = None
) -> Dict[str, List[float]]:
    """Run vanilla CFR on Leduc hold'em and return the average strategy.

    The traversal enumerates all 30 ordered hole-card deals and all 4
    possible public cards per deal (the 4 cards not in the hole), so
    the strategy table covers every information set.
    """
    _ = seed  # vanilla CFR is deterministic given the enumeration
    game = LeducGame()
    nodes: Dict[str, LeducNode] = {}

    def get_node(info_set: str) -> LeducNode:
        if info_set not in nodes:
            nodes[info_set] = LeducNode(info_set=info_set)
        return nodes[info_set]

    def cfr(state: LeducState, p0: float, p1: float) -> float:
        if game.is_terminal(state):
            return game.terminal_utility(state, 0)
        player = state.current_player
        info = state.info_set(player)
        node = get_node(info)
        legal = game.legal_actions(state)
        strategy = node.get_strategy(p0 if player == 0 else p1, legal)
        util = [0.0, 0.0, 0.0]
        node_util = 0.0
        for a in legal:
            child = game.apply_action(state, a)
            util[ACTION_INDEX[a]] = cfr(
                child,
                p0 * strategy[ACTION_INDEX[a]] if player == 0 else p0,
                p1 if player == 0 else p1 * strategy[ACTION_INDEX[a]],
            )
            node_util += strategy[ACTION_INDEX[a]] * util[ACTION_INDEX[a]]
        for a in legal:
            idx = ACTION_INDEX[a]
            regret = util[idx] - node_util
            if player == 0:
                node.regret_sum[idx] += p1 * regret
            else:
                node.regret_sum[idx] += -p0 * regret
        return node_util

    for _ in range(iterations):
        for h0, h1 in itertools.permutations(DECK, 2):
            for pub in [c for c in DECK if c not in (h0, h1)]:
                state = game.deal(hole_cards=(h0, h1), public_card=pub)
                cfr(state, 1.0, 1.0)

    return {k: v.get_average_strategy() for k, v in nodes.items()}


# ---------------------------------------------------------------------------
# Best response & exploitability
# ---------------------------------------------------------------------------


def best_response_value(
    strategy: Dict[str, List[float]], player: int
) -> float:
    """Exact expected value of the best response for ``player``.

    Uses a *template-deal simultaneous traversal*: because Leduc's
    betting tree is independent of the card deal, a single traversal of
    the template tree can carry all 120 deals at once.  Each deal ``d``
    carries the opponent's reach probability; at best-response nodes
    deals sharing an information set (same ``hole[player]``, same
    revealed public card, same history) are grouped and the maximising
    action is chosen for the whole group.  Terminal values are kept
    *unweighted* — reach multiplies in exactly once, at the root sum.
    """
    game = LeducGame()

    def info_key(state: LeducState, d: Tuple[str, str, str], p: int) -> str:
        """Mirror ``LeducState.info_set`` for deal ``d`` and player ``p``."""
        pub = d[2] if (state.public is not None and state.round == 1) else ""
        return f"{d[p]}{state.round}{pub}{state.history}"

    def deal_strategy(
        state: LeducState, d: Tuple[str, str, str], p: int, legal: List[str]
    ) -> List[float]:
        s = strategy.get(info_key(state, d, p))
        if s is None:
            return [1.0 / len(legal) if a in legal else 0.0 for a in ACTIONS]
        return s

    def rec(
        state: LeducState, reach: Dict[Tuple[str, str, str], float]
    ) -> Dict[Tuple[str, str, str], float]:
        """Return *unweighted* per-deal utilities for ``player``."""
        if game.is_terminal(state):
            return {
                d: game.terminal_utility(
                    replace(state, hole=(d[0], d[1]), public=d[2]), player
                )
                for d in reach
            }
        current = state.current_player
        legal = game.legal_actions(state)
        if current == player:
            action_vals = [
                rec(game.apply_action(state, a), reach) for a in legal
            ]
            # Group deals into information sets of the BR player.
            groups: Dict[str, List[Tuple[str, str, str]]] = {}
            for d in reach:
                groups.setdefault(info_key(state, d, player), []).append(d)
            out: Dict[Tuple[str, str, str], float] = {}
            for deals in groups.values():
                best = max(
                    range(len(legal)),
                    key=lambda i: sum(
                        action_vals[i][d] * reach[d] for d in deals
                    ),
                )
                for d in deals:
                    out[d] = action_vals[best][d]
            return out
        # Opponent follows the given strategy, per their own info set.
        out: Dict[Tuple[str, str, str], float] = {d: 0.0 for d in reach}
        for a in legal:
            ai = ACTION_INDEX[a]
            probs = {
                d: deal_strategy(state, d, current, legal)[ai] for d in reach
            }
            new_reach = {
                d: r * probs[d]
                for d, r in reach.items()
                if r * probs[d] > 0.0
            }
            if not new_reach:
                continue
            sub = rec(game.apply_action(state, a), new_reach)
            for d in new_reach:
                out[d] += probs[d] * sub[d]
        return out

    # Template deal: any three distinct cards — the betting tree shape
    # is card-independent.  All 120 real deals ride along simultaneously.
    template = game.deal(hole_cards=("Js", "Qh"), public_card="Ks")
    all_deals = [
        (h0, h1, pub)
        for h0, h1 in itertools.permutations(DECK, 2)
        for pub in [c for c in DECK if c not in (h0, h1)]
    ]
    values = rec(template, {d: 1.0 for d in all_deals})
    # Each of the 120 chance outcomes is equally likely; every opponent
    # reach weight multiplies in exactly once via the root probabilities.
    return sum(values.values()) / 120.0


def exploitability(strategy: Dict[str, List[float]]) -> float:
    """Return the exact exploitability of a Leduc strategy profile.

    ``eps = (br0 + br1) / 2`` where ``br_i`` is the exact best-response
    value for player ``i``.  At a Nash equilibrium ``eps = 0`` exactly,
    making this a sound convergence measure for CFR.
    """
    br0 = best_response_value(strategy, 0)
    br1 = best_response_value(strategy, 1)
    return (br0 + br1) / 2.0


__all__ = [
    "RANKS",
    "SUITS",
    "DECK",
    "ACTIONS",
    "ACTION_INDEX",
    "LeducState",
    "LeducGame",
    "LeducNode",
    "leduc_cfr",
    "best_response_value",
    "exploitability",
]
