import torch
import pytest

from poker_ai.rules.cfr import cfr_plus_iteration, discounted_cfr_plus_iteration


class DummyGame:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def simulate_action(self, action: int) -> float:
        return float(self.values[action])


def test_regret_pruning_and_discounting():
    num_actions = 2
    game = DummyGame(torch.tensor([1.0, -1.0]))

    # Start with positive regret on action 0 only
    cum_regret = torch.tensor([1.0, 0.0])
    cum_strategy = torch.zeros(num_actions)
    cum_regret, cum_strategy = cfr_plus_iteration(
        game, cum_regret, cum_strategy, num_actions, 1, prune_threshold=0.5
    )
    assert torch.all(cum_regret >= 0)
    # Low-regret action should remain unchanged due to pruning
    assert cum_regret[1] == pytest.approx(0.0)

    # Test discounting semantics
    cum_regret = torch.tensor([2.0, 0.0])
    cum_strategy = torch.zeros(num_actions)
    cum_regret, _ = discounted_cfr_plus_iteration(
        game, cum_regret, cum_strategy, num_actions, 1, discount=0.5
    )
    assert torch.all(cum_regret >= 0)
    # After discounting and update, cumulative regret should be half and clamped
    assert torch.allclose(cum_regret, torch.tensor([1.0, 0.0]))

