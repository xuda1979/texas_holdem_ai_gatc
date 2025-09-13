"""Baseline head-to-head evaluation tests.

The goal is to provide a minimal regression guard ensuring that our reference
strategy (a simple equilibrium for a toy game) consistently outperforms a
naïve rule-based opponent and that the computed rating is deterministic across
random seeds.
"""

from __future__ import annotations

import random

from poker_ai.evaluation.exploitability import calculate_exploitability


def test_cfr_vs_random_rulebased() -> None:
    """Trained baseline should be less exploitable than a fixed rule-based agent."""

    matrix = [[0, 1], [1, 0]]
    cfr_equilibrium = [0.5, 0.5]
    random_rule = [1.0, 0.0]

    exploitable = []
    for seed in range(3):
        random.seed(seed)
        exploitable.append(
            (
                calculate_exploitability(cfr_equilibrium, matrix),
                calculate_exploitability(random_rule, matrix),
            )
        )

    # Ratings are deterministic; first element (CFR) should always be lower
    assert len({e[0] for e in exploitable}) == 1
    assert len({e[1] for e in exploitable}) == 1
    assert exploitable[0][0] < exploitable[0][1]
