import os
import sys
from dataclasses import dataclass

import pytest
import torch

# add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.utils.action_mapping import get_legal_actions_mask  # noqa: E402


@dataclass
class DummyRules:
    big_blind: int
    player_chips: list[int]
    bets: list[int]
    current_bet: int
    previous_raise_amount: int
    pot: int


@dataclass
class DummyGame:
    rules: DummyRules


@pytest.mark.parametrize(
    "rules, player, expected",
    [
        # No bet yet: fold, check, all-in bet legal
        (
            DummyRules(
                big_blind=50,
                player_chips=[1000, 1000],
                bets=[0, 0],
                current_bet=0,
                previous_raise_amount=0,
                pot=0,
            ),
            0,
            {0, 1, 9},
        ),
        # Facing bet: fold, call and sufficiently large raises legal
        (
            DummyRules(
                big_blind=50,
                player_chips=[1000, 1000],
                bets=[0, 100],
                current_bet=100,
                previous_raise_amount=100,
                pot=150,
            ),
            0,
            {0, 2, 5, 6, 7, 8, 9},
        ),
        # Short-stacked call all-in
        (
            DummyRules(
                big_blind=50,
                player_chips=[50, 1000],
                bets=[0, 100],
                current_bet=100,
                previous_raise_amount=100,
                pot=150,
            ),
            0,
            {0, 2},
        ),
    ],
)
def test_legal_actions_match_stack_blinds_bets(
    rules: DummyRules, player: int, expected: set[int]
) -> None:
    game = DummyGame(rules)
    mask = get_legal_actions_mask(game, player, 10)
    legal = {i for i, v in enumerate(mask.tolist()) if v}
    assert legal == expected
