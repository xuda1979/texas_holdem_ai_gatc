"""
Minimal Kuhn Poker + vanilla CFR for sanity checking.
References on Kuhn Poker & CFR background:
- Wikipedia overview: https://en.wikipedia.org/wiki/Kuhn_poker
- Educational CFR walkthroughs & code: https://nn.labml.ai/cfr/kuhn/index.html
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Action = str  # 'p' (pass/check) or 'b' (bet/call)
CARDS = ["J", "Q", "K"]
ACTIONS: List[Action] = ["p", "b"]


@dataclass
class Node:
    info_set: str
    regret_sum: Dict[Action, float] = field(default_factory=lambda: {a: 0.0 for a in ACTIONS})
    strat_sum: Dict[Action, float] = field(default_factory=lambda: {a: 0.0 for a in ACTIONS})

    def strategy(self) -> Dict[Action, float]:
        """Regret-matching strategy."""
        pos = {a: max(0.0, self.regret_sum[a]) for a in ACTIONS}
        normalizer = sum(pos.values())
        if normalizer <= 1e-9:
            return {a: 1.0 / len(ACTIONS) for a in ACTIONS}
        return {a: pos[a] / normalizer for a in ACTIONS}

    def average_strategy(self) -> Dict[Action, float]:
        total = sum(self.strat_sum.values())
        if total <= 1e-12:
            return {a: 1.0 / len(ACTIONS) for a in ACTIONS}
        return {a: self.strat_sum[a] / total for a in ACTIONS}


class KuhnCFR:
    """
    Two-player Kuhn Poker CFR trainer with chance-sampling.
    History is a string like "JK" + actions (e.g., "JKbp").
    Current player is len(history) % 2 after the two chance events.
    """
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}

    def train(self, iterations: int = 10_000, seed: int = 7) -> None:
        random.seed(seed)
        util = 0.0
        for _ in range(iterations):
            # deal
            cards = CARDS[:]
            random.shuffle(cards)
            history = cards[0] + cards[1]
            util += self._cfr(history, reach_p1=1.0, reach_p2=1.0, player=0)
        # return avg utility if needed

    # --- game rules ---
    @staticmethod
    def _is_terminal(h: str) -> bool:
        if len(h) <= 2:
            return False
        # last action pass or both bet
        return h[-1] == "p" or h[-2:] == "bb"

    @staticmethod
    def _terminal_utility_p1(h: str) -> int:
        # both called: pot is 4; else check/pass: pot is 2; fold gives bettor pot of 3
        p1, p2 = h[0], h[1]
        def winner() -> int:
            # K > Q > J; use string order index
            return 1 if CARDS.index(p1) > CARDS.index(p2) else -1
        if h[-2:] == "bb":
            return 2 * winner()
        if h[-2:] == "bp":  # P2 folded to P1 bet
            return 1
        if h[-1] == "p":   # last action is pass (no bets)
            return winner()
        raise RuntimeError("non-terminal asked as terminal")

    @staticmethod
    def _player_to_act(h: str) -> int:
        # after two chance events, players alternate
        return (len(h) - 2) % 2

    # --- CFR core ---
    def _cfr(self, history: str, reach_p1: float, reach_p2: float, player: int) -> float:
        if self._is_terminal(history):
            u1 = self._terminal_utility_p1(history)
            return float(u1) if player == 0 else -float(u1)

        # info set is (card visible to current player) + action history after dealing
        card = history[player]
        info_key = card + history[2:]
        node = self.nodes.get(info_key)
        if node is None:
            node = Node(info_key)
            self.nodes[info_key] = node

        # current player strategy from regrets
        strat = node.strategy()

        # recurse for actions
        util = {}
        node_util = 0.0
        for a in ACTIONS:
            next_hist = history + a
            if player == 0:
                util[a] = self._cfr(next_hist, reach_p1 * strat[a], reach_p2, 1)
            else:
                util[a] = self._cfr(next_hist, reach_p1, reach_p2 * strat[a], 0)
            node_util += strat[a] * util[a]

        # compute regrets and update
        for a in ACTIONS:
            regret = util[a] - node_util
            if player == 0:
                node.regret_sum[a] += (reach_p2 * regret)
            else:
                node.regret_sum[a] += (reach_p1 * regret)

        # strategy accumulation for average strategy
        if player == 0:
            w = reach_p1
        else:
            w = reach_p2
        for a in ACTIONS:
            node.strat_sum[a] += w * strat[a]

        return node_util

    # convenience API
    def root_bet_probs(self) -> Tuple[float, float, float]:
        """Return average bet probability at the root for J, Q, K respectively."""
        def bet_prob_for(card: str) -> float:
            key = card  # at root, info set key is just the card
            if key not in self.nodes:
                return 0.5
            avg = self.nodes[key].average_strategy()
            return float(avg["b"])
        return (bet_prob_for("J"), bet_prob_for("Q"), bet_prob_for("K"))

