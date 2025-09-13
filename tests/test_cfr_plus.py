import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.rules.cfr import cfr_plus_iteration


class DummyGame:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.call_value = torch.arange(num_actions, dtype=torch.float)
        self.idx = 0

    def simulate_action(self, action):
        return float(action)

    def get_actual_action(self):
        a = self.idx % self.num_actions
        self.idx += 1
        return a


def test_cfr_plus_iteration_pruning():
    num_actions = 3
    game = DummyGame(num_actions)
    cum_regret = torch.zeros(num_actions)
    cum_strategy = torch.zeros(num_actions)
    cum_regret, cum_strategy = cfr_plus_iteration(
        game, cum_regret, cum_strategy, num_actions, 5, prune_threshold=0.1
    )
    assert torch.all(cum_regret >= 0)
    assert torch.sum(cum_strategy) > 0


if __name__ == "__main__":
    test_cfr_plus_iteration_pruning()
    print("test_cfr_plus_iteration_pruning passed")
