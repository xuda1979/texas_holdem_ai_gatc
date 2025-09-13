import os
import sys
import unittest
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.cli import train as run_training


class TestRunTraining(unittest.TestCase):

    @patch("poker_ai.cli.train.initialize_trainer")
    @patch("time.time")
    def test_time_based_saving_triggered(self, mock_time_time, mock_initialize_trainer):
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.save_model = MagicMock()
        mock_initialize_trainer.return_value = mock_trainer_instance

        save_interval_minutes = 1
        save_interval_seconds = save_interval_minutes * 60
        num_hands_to_simulate = 3

        # time.time() calls: 1 for init last_save_time, N for N hands, +1 for each save
        mock_time_time.side_effect = [
            1000.0,  # Call 1: Initial last_save_time
            1000.0 + 30.0,  # Call 2: Hand 1 check (30s elapsed from initial) -> no save
            1000.0
            + save_interval_seconds
            + 1.0,  # Call 3: Hand 2 check (61s elapsed from initial) -> save.
            1000.0 + save_interval_seconds + 1.0,  # Call 4: update last_save_time after save
            1000.0
            + save_interval_seconds
            + 1.0
            + 30.0,  # Call 5: Hand 3 check (30s elapsed from last_save_time) -> no save
        ]

        args_mock = MagicMock()
        args_mock.save_minutes = save_interval_minutes
        args_mock.save_model_every = 0  # Disable hand-based saving
        args_mock.num_hands = num_hands_to_simulate
        args_mock.algorithm = "ai_cfr"
        args_mock.config = "dummy_config.yaml"
        args_mock.device = None

        training_params_mock = {
            "save_model_every_minutes": save_interval_minutes,
            "save_model_every_n_hands": 0,
            "num_training_hands": num_hands_to_simulate,
        }
        game_engine_config_mock = {
            "num_players": 2,
            "starting_stack": 1000,
            "big_blind": 10,
            "small_blind": 5,
        }

        with (
            patch("poker_ai.cli.train.parse_args", return_value=args_mock),
            patch(
                "poker_ai.cli.train.load_configuration",
                return_value={
                    "training": training_params_mock,
                    "game_engine": game_engine_config_mock,
                    "model": {},
                    "curriculum": {"stages": []},
                },
            ),
            patch("poker_ai.cli.train.SelfPlay") as MockSelfPlay,
        ):  # Corrected patch target

            mock_self_play_instance = MockSelfPlay.return_value
            mock_self_play_instance.play_hand_for_training = MagicMock()

            run_training.main()

            self.assertEqual(
                mock_trainer_instance.save_model.call_count, 2
            )  # 1 conditional + 1 final

    @patch("poker_ai.cli.train.initialize_trainer")
    @patch("time.time")
    def test_hand_based_saving_still_works(self, mock_time_time, mock_initialize_trainer):
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.save_model = MagicMock()
        mock_initialize_trainer.return_value = mock_trainer_instance

        mock_time_time.return_value = 1000.0  # Time doesn't advance enough

        args_mock = MagicMock()
        args_mock.save_minutes = 0  # Disable time-based
        args_mock.save_model_every = 2  # Save every 2 hands
        args_mock.num_hands = 5
        args_mock.algorithm = "ai_cfr"
        args_mock.config = "dummy_config.yaml"
        args_mock.device = None

        training_params_mock = {
            "save_model_every_minutes": 0,
            "save_model_every_n_hands": args_mock.save_model_every,
            "num_training_hands": args_mock.num_hands,
        }
        game_engine_config_mock = {
            "num_players": 2,
            "starting_stack": 1000,
            "big_blind": 10,
            "small_blind": 5,
        }

        with (
            patch("poker_ai.cli.train.parse_args", return_value=args_mock),
            patch(
                "poker_ai.cli.train.load_configuration",
                return_value={
                    "training": training_params_mock,
                    "game_engine": game_engine_config_mock,
                    "model": {},
                    "curriculum": {"stages": []},
                },
            ),
            patch("poker_ai.cli.train.SelfPlay") as MockSelfPlay,
        ):  # Corrected patch target

            mock_self_play_instance = MockSelfPlay.return_value
            mock_self_play_instance.play_hand_for_training = MagicMock()

            run_training.main()

            self.assertEqual(
                mock_trainer_instance.save_model.call_count, 3
            )  # Saves at hand 2, 4 + 1 final

    @patch("poker_ai.cli.train.initialize_trainer")
    @patch("time.time")
    def test_no_saving_if_conditions_not_met(self, mock_time_time, mock_initialize_trainer):
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.save_model = MagicMock()
        mock_initialize_trainer.return_value = mock_trainer_instance

        mock_time_time.return_value = 1000.0  # Time doesn't advance

        args_mock = MagicMock()
        args_mock.save_minutes = 10  # High time interval
        args_mock.save_model_every = 100  # High hand interval
        args_mock.num_hands = 5
        args_mock.algorithm = "ai_cfr"
        args_mock.config = "dummy_config.yaml"
        args_mock.device = None

        training_params_mock = {
            "save_model_every_minutes": args_mock.save_minutes,
            "save_model_every_n_hands": args_mock.save_model_every,
            "num_training_hands": args_mock.num_hands,
        }
        game_engine_config_mock = {
            "num_players": 2,
            "starting_stack": 1000,
            "big_blind": 10,
            "small_blind": 5,
        }

        with (
            patch("poker_ai.cli.train.parse_args", return_value=args_mock),
            patch(
                "poker_ai.cli.train.load_configuration",
                return_value={
                    "training": training_params_mock,
                    "game_engine": game_engine_config_mock,
                    "model": {},
                    "curriculum": {"stages": []},
                },
            ),
            patch("poker_ai.cli.train.SelfPlay") as MockSelfPlay,
        ):  # Corrected patch target

            mock_self_play_instance = MockSelfPlay.return_value
            mock_self_play_instance.play_hand_for_training = MagicMock()

            run_training.main()
            self.assertEqual(
                mock_trainer_instance.save_model.call_count, 1
            )  # 0 conditional + 1 final

    @patch("poker_ai.cli.train.initialize_trainer")
    @patch("time.time")
    def test_time_based_saving_with_default_config(self, mock_time_time, mock_initialize_trainer):
        # Tests if default time (10 min) is used if not in args and specific value not in loaded config
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.save_model = MagicMock()
        mock_initialize_trainer.return_value = mock_trainer_instance

        default_save_interval_minutes = 10  # Default in run_training.py if not in config
        save_interval_seconds = default_save_interval_minutes * 60
        num_hands_to_simulate = 1

        mock_time_time.side_effect = [
            1000.0,  # Initial last_save_time
            1000.0 + save_interval_seconds + 1.0,  # Hand 1 check -> save
            1000.0 + save_interval_seconds + 1.0,  # update last_save_time after save
        ]

        args_mock = MagicMock()
        args_mock.save_minutes = None  # Simulate CLI arg not provided
        args_mock.save_model_every = 0
        args_mock.num_hands = num_hands_to_simulate
        args_mock.algorithm = "ai_cfr"
        args_mock.config = "dummy_config.yaml"
        args_mock.device = None

        # Simulate config loaded WITHOUT 'save_model_every_minutes' explicitly
        training_params_mock = {
            "save_model_every_n_hands": 0,
            "num_training_hands": num_hands_to_simulate,
            # 'save_model_every_minutes' is missing here
        }
        game_engine_config_mock = {
            "num_players": 2,
            "starting_stack": 1000,
            "big_blind": 10,
            "small_blind": 5,
        }

        with (
            patch("poker_ai.cli.train.parse_args", return_value=args_mock),
            patch(
                "poker_ai.cli.train.load_configuration",
                return_value={
                    "training": training_params_mock,
                    "game_engine": game_engine_config_mock,
                    "model": {},
                    "curriculum": {"stages": []},
                },
            ),
            patch("poker_ai.cli.train.SelfPlay") as MockSelfPlay,
        ):  # Corrected patch target

            mock_self_play_instance = MockSelfPlay.return_value
            mock_self_play_instance.play_hand_for_training = MagicMock()

            run_training.main()

            self.assertEqual(
                mock_trainer_instance.save_model.call_count, 2
            )  # 1 conditional + 1 final


if __name__ == "__main__":
    unittest.main()
