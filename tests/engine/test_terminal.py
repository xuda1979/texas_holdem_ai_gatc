import os
import sys
from typing import List

# ensure src path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from gatc_holdem.engine.rules import build_side_pots, split_winnings_with_odd_chips  # noqa: E402


def settle(initial: List[int], contrib: List[int], in_hand: List[bool], winners: List[int]):
    pots = build_side_pots(contrib, in_hand)
    payouts = {i: 0 for i in range(len(initial))}
    total = 0
    for pot in pots:
        total += pot.amount
        eligible_winners = [w for w in winners if w in pot.eligible]
        if eligible_winners:
            split = split_winnings_with_odd_chips(pot.amount, eligible_winners, dealer_index=0)
            for pid, amt in split.items():
                payouts[pid] += amt
    remaining = sum(contrib) - total
    if remaining > 0:
        active = [i for i, a in enumerate(in_hand) if a]
        if len(active) == 1:
            payouts[active[0]] += remaining
    final = [initial[i] - contrib[i] + payouts[i] for i in range(len(initial))]
    assert sum(final) == sum(initial)
    assert all(ch >= 0 for ch in final)
    return final


def test_fold_call_showdown_payouts() -> None:
    # dead blind / walk pot
    final = settle([100, 100], [1, 2], [False, True], [1])
    assert final == [99, 101]
    # player calls and loses at showdown
    final = settle([100, 100], [2, 2], [True, True], [0])
    assert final == [102, 98]
    # showdown tie
    final = settle([100, 100], [50, 50], [True, True], [0, 1])
    assert final == [100, 100]
