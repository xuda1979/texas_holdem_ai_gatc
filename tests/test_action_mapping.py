import os
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.utils.action_mapping import action_to_tuple, get_action_from_index


class DummyGameState:
    def __init__(self, pot: int, current_bet: int):
        self.pot = pot
        self.current_bet = current_bet


def test_all_indices_no_current_bet():
    gs = DummyGameState(pot=100, current_bet=0)
    for idx in range(10):
        action, amount = action_to_tuple(get_action_from_index(idx, gs, 50))
        assert isinstance(action, str)
        if idx == 0:
            assert action == "fold" and amount is None
        if idx == 1:
            assert action == "check" and amount is None
        if idx == 9:
            assert action == "raise" and amount == 50


def test_high_index_maps_to_all_in():
    gs = DummyGameState(pot=10, current_bet=0)
    action, amount = action_to_tuple(get_action_from_index(10, gs, 10))
    assert action == "raise"
    assert amount == 10
