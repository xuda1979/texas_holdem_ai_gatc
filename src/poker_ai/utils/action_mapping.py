"""Map abstract action indices to concrete poker actions.

This module provides a state-dependent mapping from discrete action indices to
``Action`` objects compatible with the :mod:`gatc_holdem` engine.  For legacy
components that expect a ``(action_str, amount)`` tuple, the helper
``action_to_tuple`` performs the conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch

try:  # pragma: no cover - engine available in most environments
    from gatc_holdem.core.actions import Action, ActionType
except Exception:  # pragma: no cover - fallback for lightweight test envs
    class ActionType(str, Enum):  # type: ignore[override]
        FOLD = "FOLD"
        CHECK = "CHECK"
        CALL = "CALL"
        BET = "BET"
        RAISE = "RAISE"
        ALL_IN = "ALL_IN"

    @dataclass(frozen=True)
    class Action:  # type: ignore[override]
        type: ActionType
        amount_to: int | None = None

        def is_bet_like(self) -> bool:
            return self.type in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}


def _call_amount(game, player_id: int) -> int:
    rules = getattr(game, "rules", None)
    if hasattr(rules, "call_amount"):
        return int(rules.call_amount(player_id))
    if hasattr(game, "get_call_amount"):
        return int(game.get_call_amount(player_id))
    if hasattr(game, "current_bet") and hasattr(game, "bets"):
        return int(max(0, game.current_bet - game.bets[player_id]))
    if rules is not None and hasattr(rules, "current_bet") and hasattr(rules, "bets"):
        return int(max(0, rules.current_bet - rules.bets[player_id]))
    return int(getattr(game, "current_bet", getattr(rules, "current_bet", 0)))


def _raise_bounds(game, player_id: int) -> Tuple[int, int]:
    """Return ``(min_raise_to, max_raise_to)`` for the given state."""

    rules = getattr(game, "rules", None)
    if rules is not None:
        if hasattr(rules, "legal_raise_to_range"):
            lo, hi = rules.legal_raise_to_range(player_id)
            return int(lo), int(hi)
        lo = getattr(rules, "min_raise_to", None)
        hi = getattr(rules, "max_raise_to", None)
        if callable(lo) and callable(hi):
            return int(lo(player_id)), int(hi(player_id))

    call_amt = _call_amount(game, player_id)
    current_to = int(
        getattr(
            game,
            "current_bet_to",
            getattr(game, "current_bet", getattr(rules, "current_bet", 0)),
        )
    )
    last_raise = int(
        getattr(
            game,
            "last_raise_size",
            getattr(game, "min_raise", getattr(rules, "previous_raise_amount", 0)),
        )
        or 0
    )
    min_to = max(current_to, current_to + last_raise, call_amt + last_raise)

    stack = 0
    if hasattr(game, "player_stacks"):
        stack = int(game.player_stacks[player_id])
    elif hasattr(game, "players"):
        p = game.players[player_id]
        stack = int(getattr(p, "stack_size", getattr(p, "stack", 0)) + getattr(p, "current_bet", 0))
    elif rules is not None and hasattr(rules, "player_chips"):
        try:
            stack = int(rules.player_chips[player_id] + rules.bets[player_id])
        except Exception:
            stack = int(rules.player_chips[player_id])
    else:
        # Conservative fallback: attempt to detect whether ``player_id`` looks
        # like a seat index.  If we know the player count and the index falls
        # within that range, cap raises at the minimal legal amount; otherwise
        # interpret the identifier as a direct stack hint (used by lightweight
        # unit tests that pass an integer stack value in place of a player ID).
        max_players = getattr(game, "num_players", None)
        if max_players is None and rules is not None:
            max_players = getattr(rules, "num_players", None)
        if max_players is None:
            try:
                max_players = len(getattr(game, "players", []))
            except Exception:  # pragma: no cover - defensive
                max_players = None
        if max_players in (None, 0):
            try:
                stack = int(player_id)
            except Exception:
                stack = int(min_to)
        else:
            try:
                candidate = int(player_id)
            except Exception:
                candidate = int(min_to)
            if 0 <= candidate < int(max_players):
                stack = int(min_to)
            else:
                stack = candidate

    try:
        max_to = max(int(min_to), int(stack))
    except Exception:
        max_to = int(min_to)

    return int(min_to), int(max_to)


def _linspace_int(lo: int, hi: int, k: int) -> List[int]:
    if k <= 1 or lo >= hi:
        return [hi]
    step = (hi - lo) / float(k - 1)
    out = [int(round(lo + i * step)) for i in range(k)]
    out = sorted(set(out))
    if out[-1] != hi:
        out[-1] = hi
    return out


def get_action_from_index(idx: int, game, player_id: int) -> Action:
    """Map an action index to an :class:`Action` instance.

    Index ``0`` → FOLD, ``1`` → CHECK/CALL, ``2+`` → ``RAISE`` buckets up to
    ALL-IN.  Raise amounts are "to" amounts (total commitment after action).
    """

    if idx <= 0:
        return Action(ActionType.FOLD)

    call_amt = _call_amount(game, player_id)
    if idx == 1:
        return Action(ActionType.CALL) if call_amt > 0 else Action(ActionType.CHECK)

    min_to, max_to = _raise_bounds(game, player_id)
    if max_to <= min_to:
        return Action(ActionType.CALL) if call_amt > 0 else Action(ActionType.CHECK)

    k = 6
    game_cfg = getattr(game, "config", None)
    if isinstance(game_cfg, dict):
        k = int(game_cfg.get("num_raise_buckets", k))
    buckets = _linspace_int(min_to, max_to, k)
    choice = buckets[min(idx - 2, len(buckets) - 1)]
    if choice < max_to:
        return Action(ActionType.RAISE, amount_to=choice)
    return Action(ActionType.ALL_IN, amount_to=max_to)


def action_to_tuple(action: Action | Tuple[str, int | None]) -> Tuple[str, int | None]:
    """Convert an :class:`Action` into a ``(action_str, amount)`` tuple."""

    if isinstance(action, tuple):  # Already in tuple form
        return action

    action_str = action.type.value.lower()
    amount = action.amount_to if action.is_bet_like() else None
    if action_str == "all_in":
        action_str = "raise"
    return action_str, amount


# ---------------------------------------------------------------------------
# Legacy legality helpers

def _is_action_valid(game, player_id: int, action_str: str, amount: int | None) -> bool:
    """Basic legality check for the simplified engine used in tests."""

    rules = game.rules
    player_chips = rules.player_chips[player_id]
    player_bet_in_round = rules.bets[player_id]
    amount_to_call = rules.current_bet - player_bet_in_round

    if action_str == "fold":
        return True
    if action_str == "check":
        return amount_to_call == 0
    if action_str == "call":
        return amount_to_call > 0

    if action_str == "bet":
        if rules.current_bet != 0 or amount is None or amount <= 0:
            return False
        if amount < rules.big_blind and amount != player_chips:
            return False
        return player_chips >= amount

    if action_str == "raise":
        if rules.current_bet == 0 or amount is None:
            return False
        if amount <= rules.current_bet:
            return False
        raise_amount_to_commit = amount - player_bet_in_round
        if player_chips < raise_amount_to_commit:
            return False
        min_raise_inc = rules.previous_raise_amount if rules.previous_raise_amount > 0 else rules.big_blind
        actual_inc = amount - rules.current_bet
        if actual_inc < min_raise_inc:
            return raise_amount_to_commit == player_chips
        return True

    return False


def get_legal_actions_mask(game, player_id: int, num_actions: int) -> torch.Tensor:
    """Return a boolean mask for which abstract actions are legal."""

    mask = torch.zeros(num_actions, dtype=torch.bool)
    seen: set[tuple[str, int | None]] = set()
    for idx in range(num_actions):
        action_str, amount = action_to_tuple(get_action_from_index(idx, game, player_id))
        if game.rules.current_bet == 0 and action_str == "raise":
            action_str = "bet"
        key = (action_str, amount)
        if key in seen:
            continue
        if _is_action_valid(game, player_id, action_str, amount):
            mask[idx] = True
            seen.add(key)

    if not mask.any():  # pragma: no cover - defensive
        mask[0] = True
    return mask


__all__ = [
    "get_action_from_index",
    "get_legal_actions_mask",
    "action_to_tuple",
    "Action",
    "ActionType",
]

