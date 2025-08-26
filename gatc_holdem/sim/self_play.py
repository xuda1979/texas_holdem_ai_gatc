from __future__ import annotations
import random
from typing import List

from gatc_holdem.engine.rules import (
    min_bet,
    min_raise_to,
    raise_reopens_action,
    build_side_pots,
    split_winnings_with_odd_chips,
)


def _demo_min_raise_scenario() -> None:
    """Demonstrate min-raise chaining & all-in reopen rule."""
    bb = 100
    assert min_bet(bb) == 100
    # Example: preflop open to 250 => last_raise_size=150 (over BB)
    current_bet_to = 250
    last_raise_size = 150
    assert min_raise_to(current_bet_to, last_raise_size, bb) == 400  # next min to
    # Raise to 400 -> next min to 550
    current_bet_to = 400
    last_raise_size = 150
    assert min_raise_to(current_bet_to, last_raise_size, bb) == 550
    # Short all-in to 500 (raise by 100) does NOT reopen
    assert not raise_reopens_action(
        raise_to=500, current_bet_to=400, last_raise_size=150
    )


def _demo_side_pots() -> None:
    """Demonstrate side-pot construction and odd-chip distribution."""
    # Stacks: 1200, 800, 500; assume all reach showdown and commit their stacks
    contrib = [1200, 800, 500]
    in_hand = [True, True, True]
    pots = build_side_pots(contrib, in_hand)
    # Expected: main 1500 among [0,1,2]; side 600 among [0,1]; extra 400 from p0 returned
    assert len(pots) == 2
    assert pots[0].amount == 1500 and set(pots[0].eligible) == {0, 1, 2}
    assert pots[1].amount == 600 and set(pots[1].eligible) == {0, 1}
    # Odd-chip example: split 5 among two winners, button at seat 1 -> seat 2 gets the odd chip
    payout = split_winnings_with_odd_chips(5, winners=[1, 2], dealer_index=1)
    assert payout[1] == 2 and payout[2] == 3


def main(argv: List[str] | None = None) -> None:
    """Tiny self-play skeleton (smoke checks only).

    This doesn't try to be a full engine; it exercises the tricky rule helpers
    you'll use from your training loop / environment.
    """
    random.seed(7)
    _demo_min_raise_scenario()
    _demo_side_pots()
    print("Self-play smoke checks passed (rules + pots).")


if __name__ == "__main__":
    main()
