"""Regression tests for CFR regret baselines in the deep trainers.

The instantaneous regret r(a) = Q(a) - V must use the state value under the
*current regret-matched policy* (the distribution self-play actually follows),
not the uniform mean of action values.  A mean baseline shifts which regrets
clamp positive and biases regret matching.
"""

from __future__ import annotations

import torch

from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer
from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer


def _make_inputs(trainer):
    hole = torch.zeros(trainer.card_feature_dim)
    community = torch.zeros(trainer.card_feature_dim)
    history = torch.zeros(1, trainer.history_feature_dim)
    return hole, community, history


def test_deep_cfr_action_value_baseline_is_regret_matched_policy(monkeypatch):
    torch.manual_seed(0)
    trainer = DeepCFRTrainer(
        input_feature_dim=4, hidden_dim=8, num_actions=3, device="cpu", buffer_capacity=8
    )
    advantages = torch.tensor([3.0, 1.0, -2.0])  # regret matching -> [0.75, 0.25, 0]
    monkeypatch.setattr(trainer, "get_advantages", lambda *a, **k: advantages.clone())

    hole, community, history = _make_inputs(trainer)
    action_values = torch.tensor([2.0, -1.0, 0.5])
    trainer.add_experience(hole, community, history, action_values=action_values, iteration=1)

    sigma = torch.tensor([0.75, 0.25, 0.0])
    expected = action_values - torch.sum(sigma * action_values)
    stored = trainer.replay_buffer.buffer[-1][3]
    assert torch.allclose(stored, expected, atol=1e-6)


def test_deep_cfr_action_value_baseline_uniform_when_no_positive_advantage(monkeypatch):
    torch.manual_seed(0)
    trainer = DeepCFRTrainer(
        input_feature_dim=4, hidden_dim=8, num_actions=3, device="cpu", buffer_capacity=8
    )
    monkeypatch.setattr(
        trainer, "get_advantages", lambda *a, **k: torch.tensor([-1.0, -2.0, -3.0])
    )

    hole, community, history = _make_inputs(trainer)
    action_values = torch.tensor([3.0, 0.0, -3.0])
    trainer.add_experience(hole, community, history, action_values=action_values, iteration=1)

    expected = action_values - action_values.mean()  # uniform sigma == mean baseline
    stored = trainer.replay_buffer.buffer[-1][3]
    assert torch.allclose(stored, expected, atol=1e-6)


def test_deep_cfr_action_value_baseline_respects_legal_mask(monkeypatch):
    torch.manual_seed(0)
    trainer = DeepCFRTrainer(
        input_feature_dim=4, hidden_dim=8, num_actions=3, device="cpu", buffer_capacity=8
    )
    monkeypatch.setattr(
        trainer, "get_advantages", lambda *a, **k: torch.tensor([1.0, 1.0, 100.0])
    )

    hole, community, history = _make_inputs(trainer)
    action_values = torch.tensor([4.0, 2.0, 99.0])
    legal = torch.tensor([True, True, False])
    trainer.add_experience(
        hole, community, history, action_values=action_values, legal_mask=legal, iteration=1
    )

    sigma = torch.tensor([0.5, 0.5, 0.0])  # illegal action masked out of the policy
    masked_values = torch.tensor([4.0, 2.0, 0.0])
    expected = masked_values - torch.sum(sigma * masked_values)
    expected[2] = 0.0  # illegal action carries no regret
    stored = trainer.replay_buffer.buffer[-1][3]
    assert torch.allclose(stored, expected, atol=1e-6)


def test_single_network_action_value_baseline_is_regret_matched_policy(monkeypatch):
    torch.manual_seed(0)
    trainer = SingleNetworkCFRTrainer(
        input_feature_dim=4, hidden_dim=8, num_actions=3, device="cpu", buffer_capacity=8
    )
    advantages = torch.tensor([1.0, 0.0, 1.0])  # regret matching -> [0.5, 0, 0.5]
    monkeypatch.setattr(trainer, "get_advantages", lambda *a, **k: advantages.clone())

    hole, community, history = _make_inputs(trainer)
    action_values = torch.tensor([2.0, -2.0, 4.0])
    trainer.add_experience(hole, community, history, action_values=action_values, iteration=1)

    sigma = torch.tensor([0.5, 0.0, 0.5])
    expected = action_values - torch.sum(sigma * action_values)
    stored = trainer.replay_buffer.buffer[-1][3]
    assert torch.allclose(stored, expected, atol=1e-6)


def test_explicit_regrets_bypass_baseline_computation(monkeypatch):
    """When self-play already provides regrets they are stored untouched."""
    torch.manual_seed(0)
    trainer = DeepCFRTrainer(
        input_feature_dim=4, hidden_dim=8, num_actions=3, device="cpu", buffer_capacity=8
    )

    def _boom(*a, **k):  # baseline path must not run
        raise AssertionError("get_advantages should not be called")

    monkeypatch.setattr(trainer, "get_advantages", _boom)

    hole, community, history = _make_inputs(trainer)
    regrets = torch.tensor([0.5, -0.25, -0.25])
    trainer.add_experience(hole, community, history, regrets=regrets, iteration=1)
    stored = trainer.replay_buffer.buffer[-1][3]
    assert torch.allclose(stored, regrets)
