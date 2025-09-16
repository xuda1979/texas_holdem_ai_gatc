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


def test_calculate_strategy_respects_mask():
    regrets = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, True])
    probs = calculate_strategy(regrets, 3, legal_actions_mask=mask)
    assert torch.allclose(probs, torch.tensor([0.25, 0.0, 0.75]))

    regrets = torch.tensor([-1.0, -5.0, -6.0])
    probs = calculate_strategy(regrets, 3, legal_actions_mask=mask)
    assert torch.allclose(probs, torch.tensor([0.5, 0.0, 0.5]))
