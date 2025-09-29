import math

import torch

from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer
from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer


def _zeros_like(trainer):
    hole = torch.zeros(trainer.card_feature_dim)
    community = torch.zeros(trainer.card_feature_dim)
    history = torch.zeros(1, trainer.history_feature_dim)
    regrets = torch.tensor([1.0, -1.0])
    return hole, community, history, regrets


def test_single_network_training_handles_zero_iteration_weights():
    torch.manual_seed(0)
    trainer = SingleNetworkCFRTrainer(
        input_feature_dim=4,
        hidden_dim=8,
        num_actions=2,
        device="cpu",
        buffer_capacity=8,
    )

    hole, community, history, regrets = _zeros_like(trainer)
    for _ in range(4):
        trainer.replay_buffer.push(hole, community, history, regrets, 0)

    loss = trainer.train(batch_size=4)
    assert math.isfinite(loss)
    assert loss >= 0.0


def test_deep_cfr_training_handles_zero_iteration_weights():
    torch.manual_seed(0)
    trainer = DeepCFRTrainer(
        input_feature_dim=4,
        hidden_dim=8,
        num_actions=2,
        device="cpu",
        buffer_capacity=8,
    )

    hole, community, history, regrets = _zeros_like(trainer)
    for _ in range(4):
        trainer.replay_buffer.push(hole, community, history, regrets, 0)

    loss = trainer.train(batch_size=4)
    assert math.isfinite(loss)
    assert loss >= 0.0
