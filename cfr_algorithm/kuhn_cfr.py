# Lightweight, self-contained CFR reference implementation on Kuhn Poker.
# This serves as a correctness sentinel for the poker AI codebase.
# It implements Vanilla CFR with regret-matching on the toy 3-card game.
# References:
# - H. W. Kuhn (1950/1951) "A simplified two-person poker" (Kuhn poker)
# - Publicly known equilibrium value for Player 1: -1/18 ≈ -0.05556
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

CARDS = ("J", "Q", "K")
RANK = {"J": 0, "Q": 1, "K": 2}


def _is_terminal(history: str) -> bool:
    # Terminal sequences:
    # "cc"   : both check -> showdown (pot=2 => ±1 payoff w.r.t. antes)
    # "bc"   : p1 bet, p2 call -> showdown (pot=4 => ±2 payoff)
    # "bf"   : p1 bet, p2 fold -> +1 for p1
    # "cbc"  : p1 check, p2 bet, p1 call -> showdown (pot=4 => ±2 payoff)
    # "cbf"  : p1 check, p2 bet, p1 fold -> -1 for p1
    return history in ("cc", "bc", "bf", "cbc", "cbf")


def _utility(history: str, cards: Tuple[str, str]) -> int:
    p1, p2 = cards
    if history == "cc":
        return 1 if RANK[p1] > RANK[p2] else -1
    if history == "bc":
        return 2 if RANK[p1] > RANK[p2] else -2
    if history == "bf":
        return 1
    if history == "cbc":
        return 2 if RANK[p1] > RANK[p2] else -2
    if history == "cbf":
        return -1
    raise ValueError(f"Unknown terminal sequence: {history!r}")


def _actions(history: str) -> Sequence[str]:
    # Whose turn? Even length -> Player 1 (index 0), odd -> Player 2 (index 1)
    if history == "":
        return ("c", "b")  # P1 at root
    if history == "c":
        return ("c", "b")  # P2 after check
    if history == "b":
        return ("f", "c")  # P2 facing bet
    if history == "cb":
        return ("f", "c")  # P1 facing bet after check
    raise ValueError(f"Invalid non-terminal history: {history!r}")


class KuhnCFR:
    """Vanilla CFR on Kuhn poker with regret-matching.

    Dependency-free and small so it can act as a drop-in 'ground-truth'
    check for CFR logic. Trains by enumerating all 6 deals per iteration
    to reduce variance.
    """

    def __init__(self) -> None:
        self._regret_sum: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        self._strategy_sum: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        self._actions_map: Dict[str, Sequence[str]] = {}

    @staticmethod
    def _player_to_act(history: str) -> int:
        return 0 if (len(history) % 2 == 0) else 1

    def _strategy(self, infoset: str, actions: Sequence[str], reach_prob: float) -> List[float]:
        """Regret-matching policy; updates average strategy weighted by reach_prob."""

        self._actions_map[infoset] = tuple(actions)
        regrets = self._regret_sum[infoset]
        pos = [max(r, 0.0) for r in regrets[: len(actions)]]
        denom = sum(pos)
        strat = [r / denom for r in pos] if denom > 0 else [1.0 / len(actions)] * len(actions)
        avg = self._strategy_sum[infoset]
        for i, p in enumerate(strat):
            avg[i] += reach_prob * p
        return strat

    def _cfr(self, cards: Tuple[str, str], history: str, p0: float, p1: float) -> float:
        """Return utility for Player 1 (index 0). Update regrets."""

        if _is_terminal(history):
            return _utility(history, cards)

        player = self._player_to_act(history)
        actions = _actions(history)
        infoset = f"{cards[player]}|{history}"
        reach = p0 if player == 0 else p1
        strat = self._strategy(infoset, actions, reach)

        child_utils, node_util = [], 0.0
        for i, a in enumerate(actions):
            next_hist = history + a
            if player == 0:
                util = self._cfr(cards, next_hist, p0 * strat[i], p1)
            else:
                util = self._cfr(cards, next_hist, p0, p1 * strat[i])
            child_utils.append(util)
            node_util += strat[i] * util

        regrets = self._regret_sum[infoset]
        opp_reach = p1 if player == 0 else p0
        for i in range(len(actions)):
            # Utilities are from P1's perspective; flip sign for P2 regrets.
            if player == 0:
                r = child_utils[i] - node_util
            else:
                r = -child_utils[i] - (-node_util)
            regrets[i] += opp_reach * r
        return node_util

    def train(self, iterations: int = 100_000) -> None:
        deals = [(c1, c2) for i, c1 in enumerate(CARDS) for j, c2 in enumerate(CARDS) if i != j]
        for _ in range(iterations):
            for d in deals:
                self._cfr(d, "", 1.0, 1.0)

    def average_strategy(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for info, sums in self._strategy_sum.items():
            actions = self._actions_map.get(info, ())
            total = sum(sums[: len(actions)])
            if total > 0:
                out[info] = [x / total for x in sums[: len(actions)]]
            else:
                out[info] = [1.0 / len(actions)] * len(actions)
        return out

    @staticmethod
    def _strategy_for(infoset: str, actions: Sequence[str], avg: Dict[str, List[float]]) -> List[float]:
        if infoset not in avg:
            return [1.0 / len(actions)] * len(actions)
        probs = avg[infoset]
        return probs[: len(actions)]

    @staticmethod
    def expected_value_p1(avg: Dict[str, List[float]]) -> float:
        """Compute EV for Player 1 under average strategies. Nash EV is -1/18 ≈ -0.05556."""

        deals = [(c1, c2) for i, c1 in enumerate(CARDS) for j, c2 in enumerate(CARDS) if i != j]
        ev = 0.0
        for c1, c2 in deals:
            deal_prob = 1.0 / 6.0
            p_c, p_b = KuhnCFR._strategy_for(f"{c1}|", ("c", "b"), avg)
            p2_c, p2_b = KuhnCFR._strategy_for(f"{c2}|c", ("c", "b"), avg)
            p2_f_b, p2_c_b = KuhnCFR._strategy_for(f"{c2}|b", ("f", "c"), avg)
            p1_f_cb, p1_c_cb = KuhnCFR._strategy_for(f"{c1}|cb", ("f", "c"), avg)
            ev += deal_prob * p_c * p2_c * (1 if RANK[c1] > RANK[c2] else -1)  # cc
            ev += deal_prob * p_c * p2_b * p1_f_cb * (-1)  # cbf
            ev += deal_prob * p_c * p2_b * p1_c_cb * (2 if RANK[c1] > RANK[c2] else -2)  # cbc
            ev += deal_prob * p_b * p2_f_b * (1)  # bf
            ev += deal_prob * p_b * p2_c_b * (2 if RANK[c1] > RANK[c2] else -2)  # bc
        return ev
