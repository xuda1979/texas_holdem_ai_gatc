from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from ..ports.engine import Engine, PlayerId
from .game_loop import EpisodeResult, GameLoop


@dataclass(frozen=True)
class SelfPlayStats:
    episodes: int
    wins: Dict[PlayerId, int]
    total_reward: Dict[PlayerId, float]
    avg_steps: float


class SelfPlayService:
    """Run many episodes and aggregate stats. Trainers and eval tools can call this."""

    def __init__(self, engine: Engine, policies: Mapping[PlayerId, object]) -> None:
        self.engine = engine
        self.policies = dict(policies)

    def run(self, n_episodes: int, seed: Optional[int] = None) -> SelfPlayStats:
        wins: Dict[PlayerId, int] = {pid: 0 for pid in range(self.engine.num_players)}
        totals: Dict[PlayerId, float] = {pid: 0.0 for pid in range(self.engine.num_players)}
        total_steps = 0

        loop = GameLoop(self.engine, self.policies)
        for i in range(n_episodes):
            s = None if seed is None else seed + i
            result: EpisodeResult = loop.play_episode(seed=s)
            total_steps += result.steps
            if result.winner is not None:
                wins[result.winner] += 1
            for pid, r in result.total_reward.items():
                totals[pid] += r

        return SelfPlayStats(
            episodes=n_episodes,
            wins=wins,
            total_reward=totals,
            avg_steps=total_steps / max(1, n_episodes),
        )
