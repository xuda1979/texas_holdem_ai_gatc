from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SidePot:
    """A side/main pot with its amount and the set of players eligible to win it."""

    amount: int
    eligible: frozenset[int]


def compute_side_pots(contrib: dict[int, int], in_showdown: set[int]) -> list[SidePot]:
    """
    Decompose total contributions into a list of pots by contribution 'levels'.
    - contrib: total chips each seat contributed this hand (including folded players)
    - in_showdown: player ids still eligible to win chips (did not fold)
    Returns pots ordered from main to highest side pot.
    Invariants:
      * sum(p.amount for p in pots) == sum(contrib.values())
      * p.eligible contains only in_showdown players who matched that level
    """
    if not contrib:
        return []

    contributions = {int(p): max(0, int(a)) for p, a in contrib.items()}

    # Build sorted positive contribution levels
    levels = sorted({a for a in contributions.values() if a > 0})
    pots: list[SidePot] = []
    prev = 0

    for lvl in levels:
        delta = lvl - prev
        if delta <= 0:
            prev = lvl
            continue

        # Eligible seats for this pot: players who reached at least this level and didn't fold
        participants = {p for p, a in contributions.items() if a >= lvl and p in in_showdown}
        if not participants:
            prev = lvl
            continue

        # Amount in this band for all contributors (including folded players' chips)
        amount = sum(max(min(a, lvl) - prev, 0) for a in contributions.values())
        if amount > 0:
            pots.append(SidePot(amount=amount, eligible=frozenset(participants)))
        prev = lvl

    return pots
