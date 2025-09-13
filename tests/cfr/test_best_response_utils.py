from poker_ai.games import KuhnTrainer, best_response_value, exploitability
from poker_ai.utils.seeding import set_seed


def uniform_strategy() -> dict[str, list[float]]:
    trainer = KuhnTrainer()
    trainer.train(1, seed=0)  # populate info sets
    return {k: [0.5, 0.5] for k in trainer.get_average_strategy()}


def test_best_response_deterministic() -> None:
    set_seed(123)
    strat = uniform_strategy()
    val1 = best_response_value(strat, 0)
    set_seed(456)
    val2 = best_response_value(strat, 0)
    assert val1 == val2


def test_exploitability_positive() -> None:
    strat = uniform_strategy()
    exp = exploitability(strat)
    assert exp > 0.0
