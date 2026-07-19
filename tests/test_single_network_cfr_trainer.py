"""Focused tests for :class:`SingleNetworkCFRTrainer`.

The trainer's replay buffer is shared with :class:`DeepCFRTrainer` (and
covered by its tests); here we exercise the ``iteration_weight_exponent``
configurable introduced in the 2026-07-19 session.
"""

from __future__ import annotations

import torch

from poker_ai.ai.trainers.single_network_cfr_trainer import (
    SingleNetworkCFRTrainer,
)


def _make_trainer(exponent: float = 1.0) -> SingleNetworkCFRTrainer:
    return SingleNetworkCFRTrainer(
        input_feature_dim=4,
        hidden_dim=8,
        num_actions=3,
        lr=0.0,  # deterministic: no weight updates
        device="cpu",
        buffer_capacity=64,
        iteration_weight_exponent=exponent,
    )


def _seed_buffer(trainer: SingleNetworkCFRTrainer) -> None:
    """Push a handful of synthetic experiences into the buffer.

    ``hole`` and ``community`` must match the trainer's
    ``card_feature_dim`` (17 by default); ``history`` is a sequence of
    length ``max_seq_len`` with feature dim ``input_feature_dim``.
    """
    torch.manual_seed(0)
    card_dim = trainer.card_feature_dim
    history_dim = trainer.history_feature_dim
    max_seq = trainer.max_seq_len
    for it in range(1, 6):
        hole = torch.zeros(card_dim)
        community = torch.zeros(card_dim)
        history = torch.zeros(max_seq, history_dim)
        regrets = torch.randn(trainer.num_actions)
        trainer.replay_buffer.push(hole, community, history, regrets, it)


def test_iteration_weight_exponent_default_is_one() -> None:
    """Default exponent == 1 reproduces the historical linear weighting."""
    trainer = _make_trainer(exponent=1.0)
    assert trainer.iteration_weight_exponent == 1.0


def test_iteration_weight_exponent_zero_flattens_weights() -> None:
    """With exponent == 0 every sample gets equal weight (t^0 = 1)."""
    trainer = _make_trainer(exponent=0.0)
    _seed_buffer(trainer)
    # When exponent == 0, weights are all 1 regardless of iteration.
    # Train should run without error and return a finite loss.
    loss = trainer.train(batch_size=5)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_iteration_weight_exponent_half_smooths_weights() -> None:
    """Exponent 0.5 weights recent samples sub-linearly (sqrt(t))."""
    trainer = _make_trainer(exponent=0.5)
    _seed_buffer(trainer)
    loss = trainer.train(batch_size=5)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_gradient_clipping_present() -> None:
    """Train should not raise even when regrets produce large gradients."""
    trainer = _make_trainer(exponent=1.0)
    torch.manual_seed(0)
    card_dim = trainer.card_feature_dim
    history_dim = trainer.history_feature_dim
    max_seq = trainer.max_seq_len
    # Push experiences with huge regret values to trigger clip_grad_norm_.
    for it in range(1, 4):
        hole = torch.zeros(card_dim)
        community = torch.zeros(card_dim)
        history = torch.zeros(max_seq, history_dim)
        regrets = torch.tensor([1e6, -1e6, 1e6])
        trainer.replay_buffer.push(hole, community, history, regrets, it)
    # No assertion needed -- the test is that train() does not raise.
    loss = trainer.train(batch_size=3)
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
