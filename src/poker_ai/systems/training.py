"""High level coordination of model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .base import Subsystem
from .cfr import CFRSubsystem
from .self_play import SelfPlaySubsystem


@dataclass(slots=True)
class TrainingController:
    """Coordinate CFR training with self-play data generation."""

    cfr: CFRSubsystem
    self_play: SelfPlaySubsystem
    iterations_per_cycle: int

    def run_cycle(self, *, start_iteration: int = 0) -> None:
        iteration = start_iteration
        for _ in range(self.iterations_per_cycle):
            self.self_play.generate_training_hand(iteration)
            iteration += 1

    def warm_start(self, num_batches: int = 1, batch_size: int = 256) -> Iterable[float]:
        for _ in range(num_batches):
            loss = self.cfr.train_from_buffer(batch_size=batch_size)
            if loss is not None:
                yield float(loss)


@dataclass(slots=True)
class TrainingSubsystem(Subsystem[TrainingController]):
    """Wrap :class:`TrainingController` so callers can register it easily."""

    @classmethod
    def create(
        cls,
        *,
        cfr: CFRSubsystem,
        self_play: SelfPlaySubsystem,
        iterations_per_cycle: int = 1,
    ) -> "TrainingSubsystem":
        controller = TrainingController(
            cfr=cfr,
            self_play=self_play,
            iterations_per_cycle=iterations_per_cycle,
        )
        subsystem = cls(name="training", component=controller)
        subsystem.inject("cfr", cfr)
        subsystem.inject("self_play", self_play)
        return subsystem
