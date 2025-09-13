import torch

from poker_ai.rules.cfr import calculate_strategy


def test_simple_rps_update():
    """Regret matching should reproduce textbook probabilities."""
    # Positive regret on rock only -> deterministic rock
    regrets = torch.tensor([1.0, -1.0, 0.0])
    probs = calculate_strategy(regrets, 3)
    assert torch.allclose(probs, torch.tensor([1.0, 0.0, 0.0]))

    # All regrets non-positive -> uniform distribution
    regrets = torch.tensor([-1.0, -2.0, -3.0])
    probs = calculate_strategy(regrets, 3)
    assert torch.allclose(probs, torch.ones(3) / 3)
