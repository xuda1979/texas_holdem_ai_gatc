import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.evaluation.performance_analysis import (
    ModelPerformanceAnalyzer,
    run_tournament,
)


class DummyTrainer:
    def __init__(self):
        self.model = AdvantageNetwork(
            history_feature_dim=18,
            card_feature_dim=18,
            hidden_dim=128,
            num_heads=4,
            num_layers=AdvantageNetwork.DEFAULT_NUM_LAYERS,
            num_actions=10,
        )

    def save_model(self, path):  # pragma: no cover - trivial
        torch.save(self.model.state_dict(), path)


class TestPerformanceAnalyzer(unittest.TestCase):
    def test_tournament_runs_after_pool_full(self):
        trainer = DummyTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ModelPerformanceAnalyzer(
                models_dir=tmpdir,
                save_every_iterations=1,
                tournament_threshold=2,
                tournament_size=2,
                games_per_match=0,
                device="cpu",
            )
            with patch(
                "poker_ai.evaluation.performance_analysis.run_tournament", return_value={}
            ) as mock_tourn:
                self.assertTrue(analyzer.on_iteration_end(trainer, 1))
                self.assertEqual(len(os.listdir(tmpdir)), 1)
                mock_tourn.assert_not_called()
                self.assertTrue(analyzer.on_iteration_end(trainer, 2))
                self.assertEqual(len(os.listdir(tmpdir)), 2)
                mock_tourn.assert_not_called()
                self.assertTrue(analyzer.on_iteration_end(trainer, 3))
                self.assertEqual(len(os.listdir(tmpdir)), 2)
                mock_tourn.assert_called_once()

    def test_disabled_when_zero_interval(self):
        trainer = DummyTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ModelPerformanceAnalyzer(
                models_dir=tmpdir,
                save_every_iterations=0,
                device="cpu",
            )
            self.assertTrue(analyzer.on_iteration_end(trainer, 1))
            self.assertEqual(os.listdir(tmpdir), [])

    def test_stop_training_after_no_improvement_samples(self):
        trainer = DummyTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ModelPerformanceAnalyzer(
                models_dir=tmpdir,
                save_every_iterations=1,
                tournament_threshold=1,
                tournament_size=1,
                games_per_match=0,
                device="cpu",
                max_no_improvement_samples=2,
            )

            existing_path = os.path.join(tmpdir, "existing.pth")
            trainer.save_model(existing_path)
            past = time.time() - 60
            os.utime(existing_path, (past, past))

            def fake_run(paths, *_args, **_kwargs):
                newest = max(paths, key=os.path.getmtime)
                return {path: (0 if path == newest else 10) for path in paths}

            with patch(
                "poker_ai.evaluation.performance_analysis.run_tournament",
                side_effect=fake_run,
            ):
                self.assertTrue(analyzer.on_iteration_end(trainer, 1))
                self.assertEqual(len(os.listdir(tmpdir)), 1)
                self.assertFalse(analyzer.on_iteration_end(trainer, 2))
                self.assertEqual(len(os.listdir(tmpdir)), 1)

    def test_run_tournament_uses_per_hand_winners(self):
        class DummyEvalStrategy:
            is_human = False

            def __init__(self, *args, **kwargs):
                pass

        class DummyGame:
            def __init__(self, num_players, starting_stack, player_strategies, verbose=False):
                self.rules = SimpleNamespace(
                    player_chips=[starting_stack for _ in range(num_players)]
                )
                self.hands_played = 0
                self.last_winner = None

            def play_game(self):
                if self.hands_played == 0:
                    self.rules.player_chips[0] += 200
                    self.rules.player_chips[1] -= 200
                    self.last_winner = 0
                elif self.hands_played == 1:
                    self.rules.player_chips[0] -= 150
                    self.rules.player_chips[1] += 150
                    self.last_winner = 1
                else:
                    self.last_winner = None
                self.hands_played += 1

        with patch(
            "poker_ai.evaluation.performance_analysis.EvalStrategy",
            DummyEvalStrategy,
        ), patch(
            "poker_ai.evaluation.performance_analysis.TexasHoldem",
            DummyGame,
        ):
            scores = run_tournament(["model_a", "model_b"], games_per_match=2)

        self.assertEqual(scores["model_a"], 1)
        self.assertEqual(scores["model_b"], 1)


if __name__ == "__main__":
    unittest.main()
