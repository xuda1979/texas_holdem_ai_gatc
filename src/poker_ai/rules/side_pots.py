from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple

"""
Robust side-pot construction for multi-way all-ins.
Inputs are *total contributions on this street* per active player.
Folded players should have contribution 0.
Returns a list of (pot_amount, eligible_player_indices) from main to deepest side pot.
"""


def _validate(contribs: Sequence[int]) -> None:
    if any(c < 0 for c in contribs):
        raise ValueError("Contributions must be non-negative")


def compute_side_pots(contribs: Sequence[int]) -> List[Tuple[int, List[int]]]:
    """
    Example:
      contribs = [100, 300, 600]  # three active players
      -> [(300, [0,1,2]), (400, [1,2]), (300, [2])]
    Invariants:
      sum(pot_amounts) == sum(contribs)
      Each pot's eligible set are players with contribution >= that level.
    """
    _validate(contribs)
    N = len(contribs)
    # Exclude zero-contribution players (folded preflop or sat out this street)
    positive = [(i, int(a)) for i, a in enumerate(contribs) if a > 0]
    if not positive:
        return []

    # Unique ascending levels
    levels = sorted({a for _, a in positive})
    pots: List[Tuple[int, List[int]]] = []
    prev = 0
    for lvl in levels:
        width = lvl - prev
        if width <= 0:
            prev = lvl
            continue
        eligible = [i for i, a in positive if a >= lvl]
        amount = width * len(eligible)
        pots.append((amount, eligible))
        prev = lvl
    # Sanity check: total equals sum of contributions
    assert sum(p for p, _ in pots) == sum(a for _, a in positive)
    return pots


def total_pot(contribs: Sequence[int]) -> int:
    """Convenience wrapper: total of all side pots."""
    return sum(a for a in contribs if a > 0)
