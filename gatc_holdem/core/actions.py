from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    BET = "BET"
    RAISE = "RAISE"
    ALL_IN = "ALL_IN"


@dataclass(frozen=True)
class Action:
    """Represents a poker action.

    For BET/RAISE/ALL_IN, `amount_to` is the *total* amount this player
    wants to have committed after the action on the current street.
    Example: if current_bet_to==250 and you want to min-raise by 150,
    you set amount_to=400.
    """

    type: ActionType
    amount_to: Optional[int] = None  # total commitment target (chips)

    def __post_init__(self) -> None:
        if self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            if self.amount_to is None:
                raise ValueError(f"{self.type} requires amount_to")
            if self.amount_to < 0:
                raise ValueError("amount_to must be non-negative")

    def is_bet_like(self) -> bool:
        return self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
