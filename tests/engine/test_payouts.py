import os
import sys
from collections import defaultdict

# ensure src in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from gatc_holdem.engine.rules import build_side_pots, split_winnings_with_odd_chips  # noqa: E402


def test_sidepots_multiway_allin() -> None:
    contrib = [101, 201, 301]
    in_hand = [True, True, True]
    pots = build_side_pots(contrib, in_hand)
    assert len(pots) == 2
    assert pots[0].amount == 303 and set(pots[0].eligible) == {0, 1, 2}
    assert pots[1].amount == 200 and set(pots[1].eligible) == {1, 2}
    dealer = 0
    payouts = defaultdict(int)
    # main pot split between players 0 and 1 with odd chip to player 1
    main_split = split_winnings_with_odd_chips(pots[0].amount, [0, 1], dealer)
    for pid, amt in main_split.items():
        payouts[pid] += amt
    # side pot split between players 1 and 2
    side_split = split_winnings_with_odd_chips(pots[1].amount, [1, 2], dealer)
    for pid, amt in side_split.items():
        payouts[pid] += amt
    # return excess contribution from player 2
    payouts[2] += contrib[2] - max(contrib[0], contrib[1])
    assert payouts == {0: 151, 1: 252, 2: 200}
    assert sum(payouts.values()) == sum(contrib)
