"""Player management utilities for the poker engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

__all__ = ["PlayerManager"]


@dataclass
class PlayerManager:
    """Track chip stacks, bets and participation for each seat."""

    player_chips: List[int]
    bets: List[int]
    total_bets_this_hand: List[int]
    active_players: List[bool]

    def update_references(
        self,
        *,
        player_chips: Optional[List[int]] = None,
        bets: Optional[List[int]] = None,
        total_bets_this_hand: Optional[List[int]] = None,
        active_players: Optional[List[bool]] = None,
    ) -> None:
        """Point the manager at freshly assigned backing lists.

        Tests sometimes replace ``rules.player_chips`` (and similar attributes)
        with entirely new lists.  The engine keeps those lists available for
        direct indexing, so the manager needs to follow the reassignment.
        """

        if player_chips is not None:
            self.player_chips = player_chips
        if bets is not None:
            self.bets = bets
        if total_bets_this_hand is not None:
            self.total_bets_this_hand = total_bets_this_hand
        if active_players is not None:
            self.active_players = active_players

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    def reset_for_new_hand(self) -> None:
        """Reset betting state while keeping chip stacks intact."""

        self.reset_bets_for_round()
        for idx, chips in enumerate(self.player_chips):
            self.total_bets_this_hand[idx] = 0
            self.active_players[idx] = chips > 0

    def reset_bets_for_round(self) -> None:
        """Clear current round bets without altering total contributions."""

        for idx in range(len(self.bets)):
            self.bets[idx] = 0

    # ------------------------------------------------------------------
    # Chip / betting operations
    # ------------------------------------------------------------------
    def post_blind(self, player_index: int, blind_amount: int) -> int:
        """Deduct a blind from ``player_index`` and return the contribution."""

        available = self.player_chips[player_index]
        contribution = min(available, blind_amount)
        self.player_chips[player_index] = max(0, available - contribution)
        self.bets[player_index] = contribution
        self.total_bets_this_hand[player_index] += contribution
        return contribution

    def apply_bet(self, player_index: int, desired_total: int) -> tuple[int, bool]:
        """Apply a bet/raise for ``player_index``.

        Returns a tuple of ``(contribution, fully_applied)`` where
        ``contribution`` is the amount moved into the pot this action and
        ``fully_applied`` signals whether the player had enough chips to reach
        ``desired_total`` (``False`` indicates an all-in for less).
        """

        current_total = self.bets[player_index]
        if desired_total < current_total:
            raise ValueError("desired_total cannot be less than the current bet")

        difference = desired_total - current_total
        if difference == 0:
            return 0, True

        available = self.player_chips[player_index]
        contribution = min(available, difference)
        self.player_chips[player_index] = max(0, available - contribution)
        self.bets[player_index] = current_total + contribution
        self.total_bets_this_hand[player_index] += contribution

        fully_applied = contribution == difference
        return contribution, fully_applied

    # ------------------------------------------------------------------
    # Player rotation helpers
    # ------------------------------------------------------------------
    def next_player_with_chips(
        self, start_index: int, *, include_start: bool = False
    ) -> Optional[int]:
        """Return the next active player with chips, wrapping around the table."""

        num_players = len(self.player_chips)
        if num_players == 0:
            return None

        idx = start_index if include_start else (start_index + 1) % num_players
        for _ in range(num_players):
            if self.active_players[idx] and self.player_chips[idx] > 0:
                return idx
            idx = (idx + 1) % num_players
        return None

    def active_with_chips(self) -> List[int]:
        """Return indices of players still able to act (active and stacked)."""

        return [
            idx
            for idx, (active, chips) in enumerate(zip(self.active_players, self.player_chips))
            if active and chips > 0
        ]

    # ------------------------------------------------------------------
    # Copy helpers
    # ------------------------------------------------------------------
    def clone(
        self,
        *,
        player_chips: List[int],
        bets: List[int],
        total_bets_this_hand: List[int],
        active_players: List[bool],
    ) -> "PlayerManager":
        """Create a clone bound to copies of the underlying lists."""

        return PlayerManager(
            player_chips=player_chips,
            bets=bets,
            total_bets_this_hand=total_bets_this_hand,
            active_players=active_players,
        )
