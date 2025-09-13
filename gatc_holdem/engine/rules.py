from __future__ import annotations

from dataclasses import dataclass


def min_bet(big_blind: int) -> int:
    """Minimum first bet on any street in NLHE is the big blind."""
    if big_blind <= 0:
        raise ValueError("big_blind must be > 0")
    return big_blind


def min_raise_to(current_bet_to: int, last_raise_size: int, big_blind: int) -> int:
    """Compute the minimum *total* amount a raise must reach.

    - If there has been no bet yet on the street (current_bet_to == 0),
      the minimum bet is the big blind.
    - Otherwise, the raise must be at least the size of the *last full raise*.
    """
    if current_bet_to < 0 or last_raise_size < 0 or big_blind <= 0:
        raise ValueError("invalid inputs")
    if current_bet_to == 0:
        return min_bet(big_blind)
    min_raise_by = max(last_raise_size, 0)
    return current_bet_to + min_raise_by


def raise_reopens_action(raise_to: int, current_bet_to: int, last_raise_size: int) -> bool:
    """Does a raise (possibly all-in) reopen the action for previous players?

    In NLHE, an *incomplete* all-in raise (size < last full raise) does NOT
    reopen the action. This returns True iff the raise size >= last full raise.
    """
    if raise_to < current_bet_to:
        raise ValueError("raise_to must be >= current_bet_to")
    raise_by = raise_to - current_bet_to
    return raise_by >= last_raise_size


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]  # player indices eligible to win this pot


def build_side_pots(contributions: list[int], in_hand: list[bool]) -> list[Pot]:
    """Build main/side pots given each player's final *street* contributions.

    Args:
      contributions: total committed by each seat to the pot this hand.
      in_hand: whether each seat is still contesting at showdown (not folded).

    Returns:
      A list of Pot(amount, eligible). Pots with <2 eligible players are omitted
      (extra chips from a lone over-contributor are effectively returned).

    Notes:
      This function is deterministic and *does not* handle odd-chip distribution
      to winners; use `split_winnings_with_odd_chips` for that.
    """
    if len(contributions) != len(in_hand):
        raise ValueError("length mismatch")
    n = len(contributions)
    if n == 0:
        return []
    # Normalize: negative contributions are illegal
    for c in contributions:
        if c < 0:
            raise ValueError("negative contribution")

    # Unique positive contribution levels
    levels = sorted({c for c in contributions if c > 0})
    if not levels:
        return []

    pots: list[Pot] = []
    prev = 0
    for level in levels:
        delta = level - prev
        eligible = [i for i, c in enumerate(contributions) if c >= level and in_hand[i]]
        if len(eligible) >= 2:
            amount = delta * len(eligible)
            pots.append(Pot(amount=amount, eligible=tuple(eligible)))
        prev = level
    return pots


def split_winnings_with_odd_chips(
    pot_amount: int, winners: list[int], dealer_index: int
) -> dict[int, int]:
    """Split a pot evenly; award odd chip(s) starting left of the button.

    Args:
      pot_amount: total chips in this specific pot.
      winners: player indices who tie for this pot.
      dealer_index: index of the dealer/button seat.

    Returns:
      dict seat->chips for this pot split.
    """
    if pot_amount < 0:
        raise ValueError("pot_amount must be >= 0")
    if not winners:
        return {}
    k = len(winners)
    base = pot_amount // k
    remainder = pot_amount % k
    # Order winners starting from first seat left of the button
    ordered = sorted(winners, key=lambda i: (i - dealer_index) % 1000000)
    payout = {w: base for w in winners}
    for i in range(remainder):
        payout[ordered[i]] += 1
    return payout
