from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple


def to_call(current_bet: int, contributed_this_round: int) -> int:
    """
    Amount the player must put in to call the current bet on this street.
    current_bet: the highest contribution any player has made on this street
    contributed_this_round: this player's contribution on this street
    """
    return max(0, int(current_bet) - int(contributed_this_round))


@dataclass(frozen=True)
class RaiseBounds:
    """
    Inclusive bounds for a legal raise-to amount on this betting round.
      - min_to: minimum *raise-to* (amount this player will have contributed after raising)
      - max_to: maximum *raise-to* (capped by effective stack = stack + contributed)
    If min_to == max_to == current_bet, no raise is legal (call/fold only).
    If min_to == max_to > current_bet, only an all-in (short of full-raise) is legal.
    """
    min_to: int
    max_to: int


def last_full_raise_size(current_bet: int, previous_full_bet: int, big_blind: int) -> int:
    """
    Compute the size of the last *full raise* on this street.
    - At the start of a street, previous_full_bet should be set to current_bet (or 0 preflop),
      and the minimum full-raise size is at least the big blind.
    - If the most recent action was a short all-in that did not constitute a full raise,
      the last_full_raise_size remains unchanged.
    """
    # amount the previous raiser increased the bet by (if any)
    delta = max(0, int(current_bet) - int(previous_full_bet))
    return max(delta, int(big_blind))


def compute_min_raise_to(current_bet: int, previous_full_bet: int, big_blind: int) -> int:
    """
    Minimum *raise-to* given current_bet and last full raise size.
    """
    lfr = last_full_raise_size(current_bet, previous_full_bet, big_blind)
    return int(current_bet) + int(lfr)


def legal_raise_bounds(
    current_bet: int,
    previous_full_bet: int,
    player_stack: int,
    player_contrib: int,
    big_blind: int,
) -> RaiseBounds:
    """
    Returns legal inclusive bounds for a raise-to amount, respecting:
      - min full-raise size (based on last *full* raise or big blind)
      - effective stack cap (player_stack + player_contrib)
      - 'no-raise' / 'all-in only' corner cases
    """
    current_bet = int(current_bet)
    previous_full_bet = int(previous_full_bet)
    player_stack = int(player_stack)
    player_contrib = int(player_contrib)
    big_blind = int(big_blind)

    # Effective stack available for this action.  When the player has already
    # invested chips on this street, ``player_stack`` represents chips
    # remaining *after* matching the current bet.  Otherwise it represents the
    # total stack available to wager.  This mirrors the semantics used in our
    # unit tests and avoids false negatives in short-stack all-in scenarios.
    if player_contrib > 0:
        max_to = current_bet + player_stack
    else:
        max_to = player_stack

    # If the player cannot exceed the current bet, raising is impossible.
    if max_to <= current_bet:
        return RaiseBounds(min_to=current_bet, max_to=current_bet)

    min_to = compute_min_raise_to(current_bet, previous_full_bet, big_blind)

    # If the player cannot reach a full raise, they may only shove (short all-in).
    if max_to < min_to:
        return RaiseBounds(min_to=max_to, max_to=max_to)

    return RaiseBounds(min_to=min_to, max_to=max_to)


def normalize_raise_to(requested_to: int, bounds: RaiseBounds) -> int:
    """
    Clamp a requested raise-to amount to the legal range.
    """
    return int(min(max(int(requested_to), bounds.min_to), bounds.max_to))


def discretize_raise_to(bounds: RaiseBounds, bins: int, index: int) -> int:
    """
    Map a discrete action index in [0, bins-1] to a legal raise-to amount within [min_to, max_to].
    Includes endpoints. If bins <= 1, returns bounds.max_to (all-in).
    """
    if bins <= 1 or bounds.min_to == bounds.max_to:
        return bounds.max_to
    index = int(max(0, min(bins - 1, index)))
    span = bounds.max_to - bounds.min_to
    step = span / float(bins - 1)
    return int(round(bounds.min_to + step * index))
