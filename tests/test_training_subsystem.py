from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from poker_ai.systems.training import TrainingController


class SpyLoggingSubsystem:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def event(self, event_name: str, **payload: object) -> None:
        self.events.append((event_name, payload))


class SpyEvaluationSubsystem:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run_tournament(self, **kwargs: object) -> dict[str, int]:
        self.calls.append(dict(kwargs))
        return {"wins": 7, "losses": 3}


class TrainingControllerTests(unittest.TestCase):
    def test_run_cycle_reports_progress_and_periodic_evaluation(self) -> None:
        cfr = MagicMock()
        self_play = MagicMock()
        self_play.generate_training_hand.side_effect = [["a", "b"], ["c"], []]
        logging_system = SpyLoggingSubsystem()
        evaluation_system = SpyEvaluationSubsystem()

        controller = TrainingController(
            cfr=cfr,
            self_play=self_play,
            iterations_per_cycle=3,
            logging=logging_system,
            evaluation=evaluation_system,
            evaluation_interval=2,
            evaluation_factory=lambda iteration: {"checkpoint": f"iter-{iteration}"},
        )

        report = controller.run_cycle(start_iteration=5)

        self.assertEqual(
            report,
            [
                {"iteration": 5, "generated_samples": 2},
                {
                    "iteration": 6,
                    "generated_samples": 1,
                    "evaluation": {"wins": 7, "losses": 3},
                },
                {"iteration": 7, "generated_samples": 0},
            ],
        )
        self.assertEqual(evaluation_system.calls, [{"checkpoint": "iter-6"}])
        self.assertEqual(
            logging_system.events,
            [
                ("training.iteration.completed", {"iteration": 5, "generated_samples": 2}),
                ("training.iteration.completed", {"iteration": 6, "generated_samples": 1}),
                (
                    "training.evaluation.completed",
                    {
                        "iteration": 6,
                        "result": {"wins": 7, "losses": 3},
                    },
                ),
                ("training.iteration.completed", {"iteration": 7, "generated_samples": 0}),
            ],
        )

    def test_warm_start_emits_loss_events_for_completed_batches(self) -> None:
        cfr = MagicMock()
        cfr.train_from_buffer.side_effect = [1.25, None, 0.5]
        self_play = MagicMock()
        logging_system = SpyLoggingSubsystem()

        controller = TrainingController(
            cfr=cfr,
            self_play=self_play,
            iterations_per_cycle=0,
            logging=logging_system,
        )

        losses = list(controller.warm_start(num_batches=3, batch_size=64))

        self.assertEqual(losses, [1.25, 0.5])
        self.assertEqual(
            logging_system.events,
            [
                (
                    "training.warm_start.batch.completed",
                    {"batch_index": 0, "batch_size": 64, "loss": 1.25},
                ),
                (
                    "training.warm_start.batch.completed",
                    {"batch_index": 2, "batch_size": 64, "loss": 0.5},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
