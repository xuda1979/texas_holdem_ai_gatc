"""High level coordination of model training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .base import Subsystem
from .cfr import CFRSubsystem
from .self_play import SelfPlaySubsystem


@dataclass
class TrainingController:
    """Coordinate CFR training with self-play data generation."""

    cfr: CFRSubsystem
    self_play: SelfPlaySubsystem
    iterations_per_cycle: int
    logging: Any | None = None
    evaluation: Any | None = None
    evaluation_interval: int | None = None
    evaluation_factory: Callable[[int], Mapping[str, Any]] | None = None

    def _emit_event(self, event_name: str, **payload: Any) -> None:
        if self.logging is None:
            return
        event = getattr(self.logging, "event", None)
        if callable(event):
            event(event_name, **payload)

    @staticmethod
    def _generated_sample_count(payload: Any) -> int | None:
        try:
            return len(payload)
        except TypeError:
            return None

    def _should_run_evaluation(self, completed_iterations: int) -> bool:
        return bool(
            self.evaluation is not None
            and self.evaluation_interval
            and completed_iterations % self.evaluation_interval == 0
        )

    def run_cycle(self, *, start_iteration: int = 0) -> list[dict[str, Any]]:
        iteration = start_iteration
        report: list[dict[str, Any]] = []
        for _ in range(self.iterations_per_cycle):
            hand_payload = self.self_play.generate_training_hand(iteration)
            iteration_report: dict[str, Any] = {"iteration": iteration}
            generated_samples = self._generated_sample_count(hand_payload)
            if generated_samples is not None:
                iteration_report["generated_samples"] = generated_samples
            self._emit_event("training.iteration.completed", **iteration_report)

            completed_iterations = iteration - start_iteration + 1
            if self._should_run_evaluation(completed_iterations):
                evaluation_kwargs: Mapping[str, Any] = {}
                if self.evaluation_factory is not None:
                    evaluation_kwargs = self.evaluation_factory(iteration)
                evaluation_result = self.evaluation.run_tournament(**dict(evaluation_kwargs))
                iteration_report["evaluation"] = dict(evaluation_result)
                self._emit_event(
                    "training.evaluation.completed",
                    iteration=iteration,
                    result=dict(evaluation_result),
                )

            report.append(iteration_report)
            iteration += 1
        return report

    def warm_start(self, num_batches: int = 1, batch_size: int = 256) -> Iterable[float]:
        for batch_index in range(num_batches):
            loss = self.cfr.train_from_buffer(batch_size=batch_size)
            if loss is not None:
                loss_value = float(loss)
                self._emit_event(
                    "training.warm_start.batch.completed",
                    batch_index=batch_index,
                    batch_size=batch_size,
                    loss=loss_value,
                )
                yield loss_value


@dataclass
class TrainingSubsystem(Subsystem[TrainingController]):
    """Wrap :class:`TrainingController` so callers can register it easily."""

    @classmethod
    def create(
        cls,
        *,
        cfr: CFRSubsystem,
        self_play: SelfPlaySubsystem,
        iterations_per_cycle: int = 1,
        logging: Any | None = None,
        evaluation: Any | None = None,
        evaluation_interval: int | None = None,
        evaluation_factory: Callable[[int], Mapping[str, Any]] | None = None,
    ) -> "TrainingSubsystem":
        controller = TrainingController(
            cfr=cfr,
            self_play=self_play,
            iterations_per_cycle=iterations_per_cycle,
            logging=logging,
            evaluation=evaluation,
            evaluation_interval=evaluation_interval,
            evaluation_factory=evaluation_factory,
        )
        subsystem = cls(name="training", component=controller)
        subsystem.inject("cfr", cfr)
        subsystem.inject("self_play", self_play)
        if logging is not None:
            subsystem.inject("logging", logging)
        if evaluation is not None:
            subsystem.inject("evaluation", evaluation)
        return subsystem
