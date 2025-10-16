"""Subsystem wrapping evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from poker_ai.evaluation import performance_analysis

from .base import Subsystem


@dataclass(slots=True)
class EvaluationManager:
    """Provide a cohesive entry point for performance evaluation."""

    tournament_runner: Callable[..., Mapping[str, int]]

    def run_tournament(self, *args, **kwargs) -> Mapping[str, int]:
        return self.tournament_runner(*args, **kwargs)


@dataclass(slots=True)
class EvaluationSubsystem(Subsystem[EvaluationManager]):
    """Construct :class:`EvaluationManager` instances."""

    @classmethod
    def create(cls) -> "EvaluationSubsystem":
        manager = EvaluationManager(performance_analysis.run_tournament)
        return cls(name="evaluation", component=manager)
