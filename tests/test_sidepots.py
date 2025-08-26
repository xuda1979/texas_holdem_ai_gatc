from __future__ import annotations
from gatc_holdem.engine.rules import build_side_pots, split_winnings_with_odd_chips


def test_3way_allin_sidepots_and_return_of_excess() -> None:
    # Stacks 1200, 800, 500
    contrib = [1200, 800, 500]
    in_hand = [True, True, True]
    pots = build_side_pots(contrib, in_hand)
    assert len(pots) == 2
    # main pot: 500 x 3
    assert pots[0].amount == 1500 and set(pots[0].eligible) == {0, 1, 2}
    # side pot: (800-500)=300 x 2 among players {0,1}
    assert pots[1].amount == 600 and set(pots[1].eligible) == {0, 1}


def test_folded_players_not_eligible_in_side_pots() -> None:
    contrib = [1000, 1000, 200]
    # Player 2 folded -> only players 0,1 eligible for all pots
    in_hand = [True, True, False]
    pots = build_side_pots(contrib, in_hand)
    # levels: 200 (eligible 0,1), 1000 (eligible 0,1) => two pots
    assert len(pots) == 2
    assert pots[0].amount == 400 and set(pots[0].eligible) == {0, 1}
    assert pots[1].amount == 1600 and set(pots[1].eligible) == {0, 1}


def test_odd_chip_goes_left_of_button_among_winners() -> None:
    # Pot of 5 among winners [1,3,4] with dealer at seat 2
    # Seat order left of button: 3,4,1 -> distribute 1 extra to 3
    payout = split_winnings_with_odd_chips(5, winners=[1, 3, 4], dealer_index=2)
    # Base: 5//3 = 1 each; remainder 2 => seats 3 and 4 get +1 in that order
    assert payout == {1: 1, 3: 2, 4: 2}
