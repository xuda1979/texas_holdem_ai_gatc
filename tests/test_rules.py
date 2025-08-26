from __future__ import annotations
import pytest
from gatc_holdem.engine.rules import min_bet, min_raise_to, raise_reopens_action


def test_min_bet_is_big_blind() -> None:
    assert min_bet(100) == 100
    with pytest.raises(ValueError):
        min_bet(0)


def test_min_raise_chain_and_allin_reopen() -> None:
    bb = 100
    # Preflop open to 250 (over BB=100) -> last_raise_size=150
    current_bet_to = 250
    last_raise_size = 150
    assert min_raise_to(current_bet_to, last_raise_size, bb) == 400
    # Next raise to 400; min next to 550
    current_bet_to = 400
    last_raise_size = 150
    assert min_raise_to(current_bet_to, last_raise_size, bb) == 550
    # All-in to 500 is +100 < 150 => does NOT reopen
    assert not raise_reopens_action(500, 400, 150)
    # But to 550 is +150 => DOES reopen
    assert raise_reopens_action(550, 400, 150)
    # Invalid: raise_to below current_bet_to
    with pytest.raises(ValueError):
        raise_reopens_action(300, 400, 150)


def test_first_bet_minimum_is_bb() -> None:
    bb = 100
    # No prior bet on street -> min bet equals BB
    assert min_raise_to(0, 0, bb) == 100
    # Even if last_raise_size unknown/0, first bet must be >= BB
    assert min_raise_to(0, 0, 50) == 50
