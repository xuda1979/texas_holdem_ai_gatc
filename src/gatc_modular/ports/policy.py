from __future__ import annotations

from typing import Protocol, Sequence

from .engine import Action, Observation, PlayerId


class Policy(Protocol):
    """An agent that chooses an action for the current player."""

    def select_action(
        self, obs: Observation, legal_actions: Sequence[Action], player_id: PlayerId
    ) -> Action:
        ...
