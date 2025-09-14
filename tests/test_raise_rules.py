import pytest
from poker_ai.rules.betting import (
    to_call,
    legal_raise_bounds,
    normalize_raise_to,
    compute_min_raise_to,
)


def test_to_call_basic():
    assert to_call(100, 0) == 100
    assert to_call(100, 40) == 60
    assert to_call(100, 120) == 0


def test_min_raise_and_all_in_only():
    # Preflop: current bet 100 (BB), last full bet 0 (start of street), BB=100
    # min raise-to = 200
    assert compute_min_raise_to(100, 0, 100) == 200

    # Player short: can only shove below min-raise
    bounds = legal_raise_bounds(
        current_bet=100,
        previous_full_bet=0,
        player_stack=50,
        player_contrib=50,  # has 50 left to get to 100, then only 50 more
        big_blind=100,
    )
    # Effective stack = 100; cannot reach 200, so only all-in to 150 is legal
    assert bounds.min_to == bounds.max_to == 150


def test_normalize_undersized_raise():
    bounds = legal_raise_bounds(
        current_bet=100,
        previous_full_bet=0,
        player_stack=1000,
        player_contrib=0,
        big_blind=100,
    )
    # min raise-to is 200; undersized 150 should clamp to 200
    assert normalize_raise_to(150, bounds) == 200


def test_full_range_is_respected():
    bounds = legal_raise_bounds(
        current_bet=300,
        previous_full_bet=100,
        player_stack=700,
        player_contrib=0,
        big_blind=100,
    )
    assert (bounds.min_to, bounds.max_to) == (500, 700)
    assert normalize_raise_to(1000, bounds) == 700
    assert normalize_raise_to(450, bounds) == 500
