import pytest
from utils.action_mapping import get_action_from_index
from game_engine.game_state import GameState


class DummyGameState:
    def __init__(self, pot: int, current_bet: int):
        self.pot = pot
        self.current_bet = current_bet


def test_all_indices_no_current_bet():
    gs = DummyGameState(pot=100, current_bet=0)
    for idx in range(10):
        action, amount = get_action_from_index(idx, gs, 50)
        assert isinstance(action, str)
        # basic sanity checks
        if idx == 0:
            assert action == "fold" and amount is None
        if idx == 1:
            assert action == "check" and amount is None
        if idx == 2:
            assert action == "call" and amount == 0
        if idx == 9:
            assert action == "raise" and amount == 50


def test_invalid_index():
    gs = DummyGameState(pot=10, current_bet=0)
    with pytest.raises(ValueError):
        get_action_from_index(10, gs, 10)
