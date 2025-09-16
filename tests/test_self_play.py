import os
import sys

from types import MethodType

import pytest
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest.mock import patch

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

    def get_advantages(self, hole, community, history):
        return torch.zeros(self.num_actions)

    def train(self, state_tensor=None, cf_payoffs=None, batch_size=None):
        return None


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
