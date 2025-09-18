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

    # Track players grouped by their exact contribution so we can update the
    # "who is still matching this level" sets without rescanning everything.
    players_by_amount: dict[int, set[int]] = {}
    positive_players: set[int] = set()
    showdown_players: set[int] = set()
    for pid, amount in contributions.items():
        if amount <= 0:
            continue
        players_by_amount.setdefault(amount, set()).add(pid)
        positive_players.add(pid)
        if pid in in_showdown:
            showdown_players.add(pid)

    if not positive_players or not showdown_players:
        return []

    # Sorted unique contribution levels ("thresholds")
    levels = sorted(players_by_amount)
    pots: list[SidePot] = []
    prev = 0
    carry = 0
    current_all = set(positive_players)
    current_showdown = set(showdown_players)

    for lvl in levels:
        removal = players_by_amount.get(prev)
        if removal:
            current_all.difference_update(removal)
            current_showdown.difference_update(removal)

        delta = lvl - prev
        if delta <= 0 or not current_all:
            prev = lvl
            continue

        band_total = delta * len(current_all)
        eligible = frozenset(current_showdown)
        if eligible:
            amount = band_total + carry
            if amount > 0:
                pots.append(SidePot(amount=amount, eligible=eligible))
            carry = 0
        else:
            carry += band_total
        prev = lvl

    if carry and pots:
        last = pots[-1]
        pots[-1] = SidePot(amount=last.amount + carry, eligible=last.eligible)

    return pots
