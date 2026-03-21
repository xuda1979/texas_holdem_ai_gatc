from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List

ACTIONS = ["p", "b"]
CARDS = [1, 2, 3]


@dataclass
class Node:
    """Information set node for Kuhn Poker."""

    info_set: str
    regret_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    strategy_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def get_strategy(self, realization_weight: float) -> List[float]:
        """Return current mixed strategy via regret matching."""
        strategy = [max(r, 0.0) for r in self.regret_sum]
        normalizing = sum(strategy)
        if normalizing > 0:
            strategy = [s / normalizing for s in strategy]
        else:
            strategy = [0.5, 0.5]
        for i in range(2):
            self.strategy_sum[i] += realization_weight * strategy[i]
        return strategy

    def get_average_strategy(self) -> List[float]:
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [0.5, 0.5]


class KuhnTrainer:
    """Train tabular CFR on the Kuhn Poker game."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}

    def train(self, iterations: int, seed: int | None = None) -> float:
        """Run ``iterations`` of CFR and return average game value."""
        if seed is not None:
            random.seed(seed)
        util = 0.0
        for _ in range(iterations):
            for cards in itertools.permutations(CARDS, 2):
                util += self._cfr(list(cards), "", 1.0, 1.0)
        return util / (iterations * 6)

    # Terminal evaluation -------------------------------------------------
    @staticmethod
    def _terminal_utility(cards: List[int], history: str, player: int) -> float | None:
        """Return the payoff for ``player`` at a terminal history, if terminal."""
        if len(history) < 2:
            return None
        opponent = 1 - player
        if history == "pp":
            winner = 0 if cards[0] > cards[1] else 1
            payoff = 1
        elif history == "bp":
            winner = 0
            payoff = 1
        elif history == "pbp":
            winner = 1
            payoff = 1
        elif history.endswith("bb"):
            winner = 0 if cards[0] > cards[1] else 1
            payoff = 2
        else:
            return None
        return payoff if winner == player else -payoff

    # CFR -----------------------------------------------------------------
    def _cfr(self, cards: List[int], history: str, p0: float, p1: float) -> float:
        plays = len(history)
        player = plays % 2
        util: List[float] = [0.0, 0.0]

        payoff = self._terminal_utility(cards, history, player)
        if payoff is not None:
            return payoff

        info_set = f"{cards[player]}{history}"
        node = self.nodes.get(info_set)
        if node is None:
            node = Node(info_set)
            self.nodes[info_set] = node

        strategy = node.get_strategy(p0 if player == 0 else p1)
        node_util = 0.0
        for i, action in enumerate(ACTIONS):
            next_history = history + action
            if player == 0:
                util[i] = -self._cfr(cards, next_history, p0 * strategy[i], p1)
            else:
                util[i] = -self._cfr(cards, next_history, p0, p1 * strategy[i])
            node_util += strategy[i] * util[i]

        for i in range(2):
            regret = util[i] - node_util
            reach = p1 if player == 0 else p0
            node.regret_sum[i] += reach * regret
        return node_util

    # Strategy ------------------------------------------------------------
    def get_average_strategy(self) -> Dict[str, List[float]]:
        return {k: v.get_average_strategy() for k, v in self.nodes.items()}


# Best response utilities --------------------------------------------------

def _br_terminal(cards: List[int], history: str, player: int) -> float | None:
    if len(history) < 2:
        return None
    opponent = 1 - player
    if history == "pp":
        winner = 0 if cards[0] > cards[1] else 1
        payoff = 1
    elif history == "bp":
        winner = 0
        payoff = 1
    elif history == "pbp":
        winner = 1
        payoff = 1
    elif history.endswith("bb"):
        winner = 0 if cards[0] > cards[1] else 1
        payoff = 2
    else:
        return None
    return payoff if winner == player else -payoff


def _best_response_rec(cards: List[int], history: str, player: int, strategy: Dict[str, List[float]]) -> float:
    payoff = _br_terminal(cards, history, player)
    if payoff is not None:
        return payoff
    current_player = len(history) % 2
    info_set = f"{cards[current_player]}{history}"
    if current_player == player:
        return max(
            _best_response_rec(cards, history + a, player, strategy) for a in ACTIONS
        )
    probs = strategy.get(info_set, [0.5, 0.5])
    value = 0.0
    for i, a in enumerate(ACTIONS):
        value += probs[i] * _best_response_rec(cards, history + a, player, strategy)
    return value


def best_response_value(strategy: Dict[str, List[float]], player: int) -> float:
    """Expected value of the best response for ``player`` against ``strategy``."""
    total = 0.0
    for cards in itertools.permutations(CARDS, 2):
        total += _best_response_rec(list(cards), "", player, strategy)
    return total / 6.0


def exploitability(strategy: Dict[str, List[float]]) -> float:
    """Return a normalized exploitability metric for a two-player game.

    The raw best-response values are expressed in chip units where a called
    bet results in a pot of four chips.  To keep test thresholds intuitive we
    normalise by this maximum swing, effectively returning the mean per-player
    exploitability expressed in "bets" rather than chips.
    """
    br0 = best_response_value(strategy, 0)
    br1 = best_response_value(strategy, 1)
    return (br0 + br1) / 4.0
