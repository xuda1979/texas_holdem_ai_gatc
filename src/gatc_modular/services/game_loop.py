from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from ..ports.engine import Action, Engine, Observation, PlayerId
from ..ports.policy import Policy


@dataclass(frozen=True)
class EpisodeResult:
    winner: Optional[PlayerId]
    total_reward: Dict[PlayerId, float]
    steps: int


class GameLoop:
    """A reusable episode runner that enforces legal actions and aggregates rewards."""

    def __init__(
        self, engine: Engine, policies: Mapping[PlayerId, Policy], enforce_legal_actions: bool = True
    ) -> None:
        self.engine = engine
        self.policies = dict(policies)
        self.enforce_legal_actions = enforce_legal_actions
        if len(self.policies) != engine.num_players:
            raise ValueError(
                f"policies ({len(self.policies)}) must match engine.num_players ({engine.num_players})"
            )

    def _ensure_legal(self, chosen: Action, legal: Sequence[Action]) -> Action:
        if chosen in legal:
            return chosen
        if not self.enforce_legal_actions:
            raise ValueError(f"Illegal action {chosen}; legal: {list(legal)}")
        # Fallback: choose the first legal action (deterministic)
        return legal[0]

    def play_episode(self, seed: Optional[int] = None, max_steps: int = 10_000) -> EpisodeResult:
        obs: Observation = self.engine.reset(seed=seed)
        rewards: Dict[PlayerId, float] = {pid: 0.0 for pid in range(self.engine.num_players)}
        steps = 0

        while not self.engine.is_terminal():
            steps += 1
            if steps > max_steps:
                raise RuntimeError(f"Episode exceeded max_steps={max_steps}")
            pid = self.engine.current_player()
            policy = self.policies.get(pid)
            if policy is None:
                raise KeyError(f"No policy provided for player {pid}")
            legal = self.engine.legal_actions()
            action = policy.select_action(obs, legal, pid)
            action = self._ensure_legal(action, legal)
            result = self.engine.step(action)
            obs = result.observation
            for p, r in result.reward.items():
                rewards[p] = rewards.get(p, 0.0) + r

        return EpisodeResult(winner=self.engine.winner(), total_reward=rewards, steps=steps)
