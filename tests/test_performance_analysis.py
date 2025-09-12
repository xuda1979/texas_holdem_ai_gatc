import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.ai.models.transformer import AdvantageNetwork  # noqa: E402
from poker_ai.evaluation.performance_analysis import ModelPerformanceAnalyzer  # noqa: E402


class DummyTrainer:
    def __init__(self: "DummyTrainer") -> None:
        self.model = AdvantageNetwork(
            history_feature_dim=18,
            card_feature_dim=17,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            num_actions=10,
        )

    def save_model(self: "DummyTrainer", path: str) -> None:  # pragma: no cover - trivial
        torch.save(self.model.state_dict(), path)


class TestPerformanceAnalyzer(unittest.TestCase):
    def test_save_and_tournament_trigger(self: "TestPerformanceAnalyzer") -> None:
        trainer = DummyTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ModelPerformanceAnalyzer(
                models_dir=tmpdir,
                save_every_samples=1,
                tournament_threshold=2,
                tournament_size=2,
                games_per_match=0,
                device="cpu",
            )
            with patch(
                "poker_ai.evaluation.performance_analysis.run_tournament", return_value={}
            ) as mock_tourn:
                analyzer.on_iteration_end(trainer, 1)
                self.assertEqual(len(os.listdir(tmpdir)), 1)
                mock_tourn.assert_not_called()
                analyzer.on_iteration_end(trainer, 2)
                self.assertEqual(len(os.listdir(tmpdir)), 2)
                mock_tourn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
