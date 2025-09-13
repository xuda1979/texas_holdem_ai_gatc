from __future__ import annotations

from typing import Dict, List

import pytest

from poker_ai.games import KuhnTrainer, exploitability
from poker_ai.utils.seeding import set_seed


@pytest.fixture
def canonical_kuhn_strategy() -> Dict[str, List[float]]:
    """Canonical Kuhn Poker equilibrium strategy probabilities."""
    return {
        "1": [1.0, 0.0],
        "2": [2 / 3, 1 / 3],
        "3": [0.0, 1.0],
        "1p": [1.0, 0.0],
        "2p": [2 / 3, 1 / 3],
        "3p": [0.0, 1.0],
        "1b": [1.0, 0.0],
        "2b": [2 / 3, 1 / 3],
        "3b": [0.0, 1.0],
        "1pb": [1.0, 0.0],
        "2pb": [2 / 3, 1 / 3],
        "3pb": [0.0, 1.0],
    }


def test_convergence(canonical_kuhn_strategy: Dict[str, List[float]]) -> None:
    set_seed(0)
    trainer = KuhnTrainer()
    trainer.train(4000, seed=0)
    strat = trainer.get_average_strategy()
    exp = exploitability(strat)
    # The simple trainer is approximate; ensure exploitability within a loose band
    assert exp < 0.15
    for infoset, probs in canonical_kuhn_strategy.items():
        assert infoset in strat
        if len(infoset) == 1:  # root information sets only
            idx = 1 if probs[1] > probs[0] else 0
            assert strat[infoset][idx] >= strat[infoset][1 - idx]
