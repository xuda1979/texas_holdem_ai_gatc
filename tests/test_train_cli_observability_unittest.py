import importlib
import json
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
for candidate in (str(SRC_PATH), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


class DummyTrainer:
    def __init__(self) -> None:
        self.config = {"training": {}}
        self.saved_models: list[str] = []
        self.replay_buffer = [1, 2, 3, 4]

    def save_model(self, path: str) -> None:
        self.saved_models.append(path)

    def train(self, batch_size: int | None = None) -> float:
        return 0.25


class DummySelfPlay:
    def __init__(self, *_, **__):
        self.played_iterations: list[int] = []

    def play_hand_for_training(self, iteration: int) -> None:
        self.played_iterations.append(iteration)


class DummyAnalyzer:
    def __init__(self, *_, **__):
        self.no_improvement_samples = 7
        self.calls: list[int] = []

    def on_iteration_end(self, trainer, iteration: int) -> bool:
        self.calls.append(iteration)
        return True


class TrainCliObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._install_dependency_stubs()
        cls.train = importlib.import_module("poker_ai.cli.train")

    @staticmethod
    def _install_dependency_stubs() -> None:
        torch_module = types.ModuleType("torch")
        torch_module.zeros = lambda *args, **kwargs: [0]
        torch_module.bool = bool
        torch_module.no_grad = lambda: (lambda fn: fn)
        torch_module.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        torch_module.npu = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        torch_module.nn = SimpleNamespace(
            Module=type("Module", (), {}),
            DataParallel=lambda module, **kwargs: module,
        )
        sys.modules["torch"] = torch_module

        transformer = types.ModuleType("poker_ai.ai.models.transformer")
        transformer.AdvantageNetwork = type(
            "AdvantageNetwork",
            (),
            {"DEFAULT_HIDDEN_DIM": 64, "DEFAULT_NUM_LAYERS": 2},
        )
        sys.modules["poker_ai.ai.models.transformer"] = transformer

        config_mod = types.ModuleType("poker_ai.config")
        config_mod.load_config = lambda path=None: {}
        sys.modules["poker_ai.config"] = config_mod

        eval_mod = types.ModuleType("poker_ai.evaluation.performance_analysis")
        eval_mod.ModelPerformanceAnalyzer = DummyAnalyzer
        sys.modules["poker_ai.evaluation.performance_analysis"] = eval_mod

        logging_utils = types.ModuleType("poker_ai.logging_utils")
        logging_utils.setup_logging = lambda *args, **kwargs: logging.getLogger("train-test")
        logging_utils.log_run_metadata = lambda *args, **kwargs: None
        logging_utils.log_configuration_snapshot = lambda *args, **kwargs: None
        sys.modules["poker_ai.logging_utils"] = logging_utils

        model_storage = types.ModuleType("poker_ai.model_storage")
        model_storage.remote_checkpoint_dir = lambda: PROJECT_ROOT / "tmp_models"
        model_storage.remote_algorithm_checkpoint_path = (
            lambda algorithm, label: PROJECT_ROOT / "tmp_models" / f"{algorithm}_{label}.pth"
        )
        sys.modules["poker_ai.model_storage"] = model_storage

        self_play_mod = types.ModuleType("poker_ai.selfplay.self_play")
        self_play_mod.SelfPlay = DummySelfPlay
        sys.modules["poker_ai.selfplay.self_play"] = self_play_mod

    def _base_config(self) -> dict:
        return {
            "training": {
                "num_training_hands": 2,
                "save_model_every_samples": 0,
                "save_model_every_n_hands": 0,
                "save_model_every_minutes": 0,
                "min_buffer_before_train": 2,
                "samples_per_cycle": 2,
                "train_steps_per_cycle": 1,
                "train_batch_size": 2,
                "max_samples": 2,
            },
            "game_engine": {
                "min_players": 2,
                "max_players": 2,
                "starting_stack": 1000,
                "big_blind": 10,
                "small_blind": 5,
            },
            "curriculum": {"stages": []},
            "model": {},
            "logging": {},
        }

    def _args(self):
        return SimpleNamespace(
            algorithm="deep_cfr",
            gpus=False,
            npus=False,
            tpu=False,
            num_hands=None,
            save_model_every=None,
            save_minutes=None,
            save_samples=None,
            min_buffer_before_train=None,
            samples_per_cycle=None,
            train_steps_per_cycle=None,
            train_batch_size=None,
            max_samples=None,
            config=None,
        )

    def test_emit_training_event_logs_structured_json(self):
        logger = logging.getLogger("train-event-test")
        messages: list[str] = []

        with patch.object(self.train, "_safe_log", side_effect=lambda *_args, **_kwargs: messages.append(_args[2] % _args[3:])):
            self.train._emit_training_event(
                logger,
                "cycle_complete",
                cycle=3,
                total_samples=128,
                avg_loss=0.125,
            )

        self.assertEqual(len(messages), 1)
        prefix, payload = messages[0].split(": ", 1)
        self.assertEqual(prefix, "Training event")
        parsed = json.loads(payload)
        self.assertEqual(parsed["event"], "cycle_complete")
        self.assertEqual(parsed["cycle"], 3)
        self.assertEqual(parsed["total_samples"], 128)
        self.assertEqual(parsed["avg_loss"], 0.125)

    def test_main_emits_cycle_evaluation_and_final_events(self):
        trainer = DummyTrainer()
        recorded_events: list[dict] = []

        def fake_initialize(*_args, **_kwargs):
            return trainer

        def capture_event(logger, event, **payload):
            recorded_events.append({"event": event, **payload})

        with (
            patch.object(self.train, "parse_args", return_value=self._args()),
            patch.object(self.train, "load_configuration", return_value=self._base_config()),
            patch.object(self.train, "initialize_trainer", side_effect=fake_initialize),
            patch.object(self.train, "_load_latest_model", return_value=False),
            patch.object(self.train, "_emit_training_event", side_effect=capture_event),
        ):
            self.train.main()

        event_names = [item["event"] for item in recorded_events]
        self.assertIn("cycle_complete", event_names)
        self.assertIn("evaluation_status", event_names)
        self.assertIn("final_model_saved", event_names)

        evaluation_event = next(item for item in recorded_events if item["event"] == "evaluation_status")
        self.assertEqual(evaluation_event["total_samples"], 2)
        self.assertEqual(evaluation_event["no_improvement_samples"], 7)
        self.assertTrue(evaluation_event["should_continue"])

        final_event = next(item for item in recorded_events if item["event"] == "final_model_saved")
        self.assertTrue(str(final_event["path"]).endswith("deep_cfr_final.pth"))


if __name__ == "__main__":
    unittest.main()
