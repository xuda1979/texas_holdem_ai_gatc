from __future__ import annotations

import pytest
import torch

from poker_ai.rules.cfr import compute_regrets


def test_node_values_match_handwritten_tree():
    """CFVs and regrets should match manual computation on toy tree."""
    # Toy tree: two actions; action values come from averaging chance outcomes
    action_values = torch.tensor([0.0, 1.0])  # expected payoffs for actions A and B
    strategy = torch.tensor([0.5, 0.5])
    state_value = torch.sum(strategy * action_values)
    assert state_value == pytest.approx(0.5)

    regrets = compute_regrets(action_values, state_value)
    assert torch.allclose(regrets, torch.tensor([-0.5, 0.5]))

