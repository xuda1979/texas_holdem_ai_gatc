import torch

from poker_ai.rules.cfr import update_strategy, compute_average_strategy


def test_weighted_accumulation_vs_hand_counts():
    weights = [1.0, 0.5, 0.25]
    strategies = [
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.2, 0.8]),
        torch.tensor([0.1, 0.9]),
    ]
    cumulative = torch.zeros(2)
    for w, strat in zip(weights, strategies):
        cumulative = update_strategy(cumulative, strat * w)
    avg = compute_average_strategy(cumulative)
    manual = (
        weights[0] * strategies[0]
        + weights[1] * strategies[1]
        + weights[2] * strategies[2]
    ) / sum(weights)
    assert torch.allclose(avg, manual)

