import math

import pytest
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


def _populate_deep_cfr_buffer(trainer: DeepCFRTrainer, *, entries: int = 6) -> None:
    for i in range(entries):
        hole = torch.full((trainer.card_feature_dim,), float(i))
        community = torch.full((trainer.card_feature_dim,), float(i) / 10.0)
        history = torch.full((1, trainer.history_feature_dim), float(i + 1))
        regrets = torch.tensor([float(i + 1), -float(i + 1)])
        strategy = torch.softmax(torch.tensor([float(i + 1), float(entries - i)]), dim=-1)
        trainer.replay_buffer.push(hole, community, history, regrets, i + 1)
        trainer.strategy_buffer.push(hole, community, history, strategy, i + 1)


def _fixed_buffer_sample(trainer: DeepCFRTrainer):
    fixed_batch = list(trainer.replay_buffer.buffer)
    fixed_strategy_batch = list(trainer.strategy_buffer.buffer)
    trainer.replay_buffer.sample = lambda batch_size: fixed_batch[:batch_size]
    trainer.strategy_buffer.sample = lambda batch_size: fixed_strategy_batch[:batch_size]


def test_deep_cfr_sharded_advantage_step_matches_single_device():
    torch.manual_seed(0)
    single = DeepCFRTrainer(
        input_feature_dim=4,
        hidden_dim=8,
        num_actions=2,
        device="cpu",
        buffer_capacity=8,
    )
    _populate_deep_cfr_buffer(single)
    _fixed_buffer_sample(single)
    single.advantage_net.eval()

    torch.manual_seed(0)
    sharded = DeepCFRTrainer(
        input_feature_dim=4,
        hidden_dim=8,
        num_actions=2,
        device="cpu",
        buffer_capacity=8,
    )
    sharded.advantage_net.load_state_dict(single.advantage_net.state_dict())
    sharded.optimizer.load_state_dict(single.optimizer.state_dict())
    _populate_deep_cfr_buffer(sharded)
    _fixed_buffer_sample(sharded)
    sharded._parallel_train_devices = ["cpu", "cpu", "cpu"]
    sharded.advantage_net.eval()

    single_loss = single.train(batch_size=6)
    sharded_loss = sharded.train(batch_size=6)

    assert sharded_loss == pytest.approx(single_loss, rel=1e-6, abs=1e-6)
    for single_param, sharded_param in zip(
        single.advantage_net.parameters(), sharded.advantage_net.parameters()
    ):
        assert torch.allclose(single_param, sharded_param, atol=1e-6, rtol=1e-6)


def test_deep_cfr_sharded_policy_step_matches_single_device():
    torch.manual_seed(0)
    single = DeepCFRTrainer(
        input_feature_dim=4,
        hidden_dim=8,
        num_actions=2,
        device="cpu",
        buffer_capacity=8,
    )
    _populate_deep_cfr_buffer(single)
    _fixed_buffer_sample(single)
    single.policy_net.eval()

    torch.manual_seed(0)
    sharded = DeepCFRTrainer(
        input_feature_dim=4,
        hidden_dim=8,
        num_actions=2,
        device="cpu",
        buffer_capacity=8,
    )
    sharded.policy_net.load_state_dict(single.policy_net.state_dict())
    sharded.policy_optimizer.load_state_dict(single.policy_optimizer.state_dict())
    _populate_deep_cfr_buffer(sharded)
    _fixed_buffer_sample(sharded)
    sharded._parallel_train_devices = ["cpu", "cpu"]
    sharded.policy_net.eval()

    single_loss = single.train_policy(batch_size=6)
    sharded_loss = sharded.train_policy(batch_size=6)

    assert sharded_loss == pytest.approx(single_loss, rel=1e-6, abs=1e-6)
    for single_param, sharded_param in zip(
        single.policy_net.parameters(), sharded.policy_net.parameters()
    ):
        assert torch.allclose(single_param, sharded_param, atol=1e-6, rtol=1e-6)
