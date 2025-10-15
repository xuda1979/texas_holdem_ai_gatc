"""Betting state helpers for the poker engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["BettingRoundTracker"]


@dataclass
class BettingRoundTracker:
    """Keep track of betting round metadata."""

    current_bet: int = 0
    previous_raise_amount: int = 0
    actions_this_round: int = 0
    last_raiser: Optional[int] = None

    def reset_round(self) -> None:
        """Reset per-round counters to their defaults."""

        self.current_bet = 0
        self.previous_raise_amount = 0
        self.actions_this_round = 0
        self.last_raiser = None

    def record_action(self) -> None:
        """Increment the action counter for the active betting round."""

        self.actions_this_round += 1

    def note_raise(self, raise_to: int, *, previous_bet: int) -> None:
        """Update bookkeeping after a player raises to ``raise_to`` chips."""

        self.previous_raise_amount = raise_to - previous_bet
        self.current_bet = raise_to

    def clone(self) -> "BettingRoundTracker":
        """Return a copy of the tracker."""

        return BettingRoundTracker(
            current_bet=self.current_bet,
            previous_raise_amount=self.previous_raise_amount,
            actions_this_round=self.actions_this_round,
            last_raiser=self.last_raiser,
        )
