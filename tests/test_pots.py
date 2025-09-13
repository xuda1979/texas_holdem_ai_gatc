import pytest

pytest.importorskip("eval7")

from gatc_poker.pots import SidePot, compute_side_pots


def amounts(pots: list[SidePot]) -> list[int]:
    return [p.amount for p in pots]


def eligibles(pots: list[SidePot]) -> list[set[int]]:
    return [set(p.eligible) for p in pots]


def test_simple_equal_contributions() -> None:
    contrib = {0: 100, 1: 100, 2: 100}
    showdown = {0, 1, 2}
    pots = compute_side_pots(contrib, showdown)
    assert amounts(pots) == [300]
    assert eligibles(pots) == [{0, 1, 2}]


def test_all_in_side_pots_everyone_live() -> None:
    # A:1000, B:2000, C:3000 (all eligible)
    contrib = {0: 1000, 1: 2000, 2: 3000}
    showdown = {0, 1, 2}
    pots = compute_side_pots(contrib, showdown)
    # base 3*1000 = 3000, then two-band 2*1000 = 2000, last 1*1000 = 1000
    assert amounts(pots) == [3000, 2000, 1000]
    assert eligibles(pots) == [{0, 1, 2}, {1, 2}, {2}]


def test_side_pots_with_folded_contributor() -> None:
    # Player 0 contributed 100 then folded; only 1 and 2 eligible at showdown
    contrib = {0: 100, 1: 200, 2: 400}
    showdown = {1, 2}
    pots = compute_side_pots(contrib, showdown)
    # Pot bands: [0-100]: 3*100 = 300 (eligible {1,2})
    #            [100-200]: 2*100 = 200 (eligible {1,2})
    #            [200-400]: 1*200 = 200 (eligible {2})
    assert amounts(pots) == [300, 200, 200]
    assert eligibles(pots) == [{1, 2}, {1, 2}, {2}]
