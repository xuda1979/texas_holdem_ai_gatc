import random

import numpy as np
import torch

from poker_ai.utils.seeding import set_seed

ACTIONS = ["fold", "call", "raise"]


def generate_action_sequence(length: int = 5) -> list[str]:
    """Generate a pseudo-random action sequence using random, numpy, and torch."""
    sequence: list[str] = []
    for _ in range(length):
        idx = (
            int(random.random() * len(ACTIONS))
            + int(np.random.rand() * len(ACTIONS))
            + int(torch.rand(1).item() * len(ACTIONS))
        ) % len(ACTIONS)
        sequence.append(ACTIONS[idx])
    return sequence


def test_action_sequence_reproducible() -> None:
    set_seed(123)
    seq1 = generate_action_sequence()
    set_seed(123)
    seq2 = generate_action_sequence()
    assert seq1 == seq2


def test_action_sequence_varies_with_seed() -> None:
    set_seed(123)
    seq1 = generate_action_sequence()
    set_seed(456)
    seq2 = generate_action_sequence()
    assert seq1 != seq2
