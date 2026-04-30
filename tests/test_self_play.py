from __future__ import annotations

import os
import sys
import time
from types import MethodType, SimpleNamespace

import pytest
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest.mock import MagicMock, patch

from poker_ai.cli.self_play import extract_game_data
from poker_ai.selfplay.self_play import SelfPlay


class DummyModel:
    num_actions = 2

    def __call__(self, x):
        return torch.zeros(self.num_actions)


class DummyTrainer:
    class Buffer(list):
        def push(self, *args):
            self.append(args)

    def __init__(self):
        self.model = DummyModel()
        self.config = {"model": {"max_seq_len": 10, "d_raw_feature": 2}}
        self.num_actions = 2
        self.replay_buffer = self.Buffer()
        self.loaded_paths: list[str | None] = []

    def get_advantages(self, hole, community, history):
        return torch.zeros(self.num_actions)

    def train(self, state_tensor=None, cf_payoffs=None, batch_size=None):
        return None

    def load_model(self, path: str | None = None):
        self.loaded_paths.append(path)


def test_play_hand_for_training_runs():
    trainer = DummyTrainer()
    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 50})
    with (
        patch.object(
            SelfPlay,
            "_get_policy",
            return_value=torch.ones(trainer.num_actions) / trainer.num_actions,
        ),
        patch(
            "poker_ai.selfplay.self_play.prepare_transformer_input",
            return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        ),
    ):
        data = sp.play_hand_for_training()
    assert isinstance(data, list)


def test_play_hand_for_training_respects_custom_buffer_threshold():
    trainer = DummyTrainer()
    trainer.replay_buffer.extend([object(), object()])
    trainer.train = MagicMock(return_value=None)

    training_cfg = {"min_buffer_before_train": 2}
    sp = SelfPlay(
        trainer,
        {"num_players": 2, "starting_stack": 50},
        training_config=training_cfg,
    )

    class DummyGame:
        def __init__(self, num_players: int, starting_stack: int) -> None:
            self.rules = SimpleNamespace(big_blind=0, small_blind=0, num_players=num_players)

        def initialize_game(self) -> None:
            pass

        def clone(self) -> "DummyGame":
            return self

    with (
        patch("poker_ai.selfplay.self_play.random.randint", return_value=2),
        patch("poker_ai.selfplay.self_play.TexasHoldem", DummyGame),
        patch.object(SelfPlay, "_traverse_mccfr", return_value=0),
    ):
        sp.play_hand_for_training(iteration=123)

    trainer.train.assert_called_once_with(batch_size=2)


def test_play_hand_for_training_trains_policy_when_strategy_buffer_ready():
    trainer = DummyTrainer()
    trainer.replay_buffer.extend([object(), object()])
    trainer.strategy_buffer = DummyTrainer.Buffer()
    trainer.strategy_buffer.extend([object(), object()])
    trainer.train = MagicMock(return_value=0.5)
    trainer.train_policy = MagicMock(return_value=0.25)

    training_cfg = {"min_buffer_before_train": 2}
    sp = SelfPlay(
        trainer,
        {"num_players": 2, "starting_stack": 50},
        training_config=training_cfg,
    )

    class DummyGame:
        def __init__(self, num_players: int, starting_stack: int) -> None:
            self.rules = SimpleNamespace(big_blind=0, small_blind=0, num_players=num_players)

        def initialize_game(self) -> None:
            pass

        def clone(self) -> "DummyGame":
            return self

    with (
        patch("poker_ai.selfplay.self_play.random.randint", return_value=2),
        patch("poker_ai.selfplay.self_play.TexasHoldem", DummyGame),
        patch.object(SelfPlay, "_traverse_mccfr", return_value=0),
    ):
        sp.play_hand_for_training(iteration=123)

    trainer.train.assert_called_once_with(batch_size=2)
    trainer.train_policy.assert_called_once_with(batch_size=2)


def test_play_hand_for_training_skips_policy_until_strategy_buffer_threshold():
    trainer = DummyTrainer()
    trainer.replay_buffer.extend([object(), object()])
    trainer.strategy_buffer = DummyTrainer.Buffer()
    trainer.strategy_buffer.append(object())
    trainer.train = MagicMock(return_value=0.5)
    trainer.train_policy = MagicMock(return_value=0.25)

    training_cfg = {"min_buffer_before_train": 2}
    sp = SelfPlay(
        trainer,
        {"num_players": 2, "starting_stack": 50},
        training_config=training_cfg,
    )

    class DummyGame:
        def __init__(self, num_players: int, starting_stack: int) -> None:
            self.rules = SimpleNamespace(big_blind=0, small_blind=0, num_players=num_players)

        def initialize_game(self) -> None:
            pass

        def clone(self) -> "DummyGame":
            return self

    with (
        patch("poker_ai.selfplay.self_play.random.randint", return_value=2),
        patch("poker_ai.selfplay.self_play.TexasHoldem", DummyGame),
        patch.object(SelfPlay, "_traverse_mccfr", return_value=0),
    ):
        sp.play_hand_for_training(iteration=123)

    trainer.train.assert_called_once_with(batch_size=2)
    trainer.train_policy.assert_not_called()


def test_self_play_reloads_latest_checkpoint_when_updated(tmp_path):
    trainer = DummyTrainer()
    trainer.train = MagicMock(return_value=None)
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"initial")

    training_cfg = {
        "save_model_path": str(model_path),
        "reload_model_every_hands": 1,
        "min_buffer_before_train": 1,
    }

    class DummyGame:
        def __init__(self, num_players: int, starting_stack: int) -> None:
            self.rules = SimpleNamespace(
                big_blind=0,
                small_blind=0,
                num_players=num_players,
                dealer_button=0,
                active_players=[True] * num_players,
                player_chips=[starting_stack] * num_players,
                community_cards=[],
                current_player=0,
            )

        def initialize_game(self) -> None:
            return None

        def clone(self) -> "DummyGame":
            return self

    with (
        patch("poker_ai.selfplay.self_play.random.randint", return_value=2),
        patch("poker_ai.selfplay.self_play.TexasHoldem", DummyGame),
        patch.object(SelfPlay, "_traverse_mccfr", return_value=0),
    ):
        sp = SelfPlay(
            trainer,
            {"num_players": 2, "starting_stack": 50},
            training_config=training_cfg,
        )

        assert trainer.loaded_paths[-1] == str(model_path)
        trainer.loaded_paths.clear()

        time.sleep(0.01)
        model_path.write_bytes(b"updated")

        sp.play_hand_for_training(iteration=1)

    assert trainer.loaded_paths[-1] == str(model_path)


def test_extract_game_data_uses_betting_history_directly():
    betting_history = [("0", ("bet", 100)), ("1", ("call", 100))]
    community_cards = ["Ah", "Kd", "Qs"]
    players = [{"stack": 900}, {"stack": 1100}]

    game = SimpleNamespace(
        hand_count=7,
        rules=SimpleNamespace(
            dealer_button=1,
            betting_history=betting_history,
            community_cards=community_cards,
            pot=200,
        ),
        get_player_status=lambda: players,
    )

    data = extract_game_data(game)

    assert data["actions"] is betting_history
    assert data["actions"] == betting_history
    assert data["community_cards"] == community_cards
    assert data["players"] == players


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_policy_handles_cuda_advantages():
    trainer = DummyTrainer()

    trainer.get_advantages = MethodType(
        lambda self, *args, **kwargs: torch.tensor([1.0, -1.0], device="cuda"),
        trainer,
    )

    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 50})

    with (
        patch(
            "poker_ai.selfplay.self_play.prepare_transformer_input",
            return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        ),
        patch(
            "poker_ai.selfplay.self_play.get_legal_actions_mask",
            return_value=torch.tensor([True, False]),
        ),
    ):
        policy = sp._get_policy(object(), 0)

    assert policy.device.type == "cuda"
