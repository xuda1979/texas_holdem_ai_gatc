from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..ports.engine import Action, Engine, Observation, PlayerId, StepResult


class CountingGameEngine(Engine):
    """
    A tiny deterministic 2-player environment used in tests.
    Players take turns choosing an action in {0,1}.
    The game lasts `horizon` steps. Reward is +1 to the player who chose action 1 on the
    final step; 0 otherwise. Winner is that player; otherwise None.
    """

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon
        self.t = 0
        self._cur = 0
        self._winner: Optional[int] = None
        self.num_players = 2

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self._cur = 0
        self._winner = None
        return (self.t, self._cur)

    def current_player(self) -> PlayerId:
        return self._cur

    def legal_actions(self) -> List[Action]:
        return [0, 1]

    def is_terminal(self) -> bool:
        return self.t >= self.horizon

    def step(self, action: Action) -> StepResult:
        if action not in (0, 1):
            raise ValueError("illegal action")
        self.t += 1
        last_actor = self._cur
        self._cur = 1 - self._cur
        reward: Dict[int, float] = {0: 0.0, 1: 0.0}
        if self.t >= self.horizon:
            if action == 1:
                reward[last_actor] = 1.0
                self._winner = last_actor
            else:
                self._winner = None
        obs: Observation = (self.t, self._cur)
        return StepResult(
            observation=obs,
            reward=reward,
            done=self.is_terminal(),
            info={"last_actor": last_actor, "action": action},
        )

    def winner(self) -> Optional[PlayerId]:
        return self._winner

    def clone(self) -> "CountingGameEngine":
        c = CountingGameEngine(self.horizon)
        c.t = self.t
        c._cur = self._cur
        c._winner = self._winner
        return c


class FirstLegalPolicy:
    def select_action(self, obs, legal_actions, player_id) -> int:  # type: ignore[override]
        return legal_actions[0]


class RandomPolicy:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, obs, legal_actions, player_id) -> int:  # type: ignore[override]
        return self._rng.choice(list(legal_actions))
