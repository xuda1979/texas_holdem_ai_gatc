from __future__ import annotations

from pathlib import Path

import torch

from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer
from poker_ai.cli import play_vs_ai


def test_cli_strategy_uses_loader_metadata(monkeypatch, tmp_path: Path) -> None:
    trainer = SingleNetworkCFRTrainer(
        input_feature_dim=21,
        hidden_dim=32,
        num_actions=5,
        lr=1e-3,
        device="cpu",
    )
    model_path = tmp_path / "trainer_strategy.pth"
    trainer.save_model(str(model_path))

    strategy = play_vs_ai.AIStrategy(str(model_path), device="cpu")

    assert strategy.model is not None
    assert strategy.feature_dim == trainer.history_feature_dim
    assert strategy.max_seq_len == trainer.max_seq_len
    assert strategy.num_actions == trainer.num_actions

    captured: dict[str, int] = {}

    def fake_prepare(game, player_index, max_seq_len, feature_dim):
        captured["max_seq_len"] = max_seq_len
        captured["feature_dim"] = feature_dim
        card_dim = trainer.card_feature_dim
        hole = torch.zeros(card_dim, dtype=torch.float32)
        community = torch.zeros(card_dim, dtype=torch.float32)
        history = torch.zeros(max_seq_len, feature_dim, dtype=torch.float32)
        return hole, community, history

    monkeypatch.setattr(play_vs_ai, "prepare_transformer_input", fake_prepare)
    monkeypatch.setattr(
        play_vs_ai,
        "get_legal_actions_mask",
        lambda game, player_index, num_actions: torch.ones(num_actions, dtype=torch.bool),
    )

    def fake_strategy(advantages, num_actions):
        return advantages.new_full((num_actions,), 1.0 / num_actions)

    monkeypatch.setattr(play_vs_ai, "calculate_strategy", fake_strategy)
    monkeypatch.setattr(
        play_vs_ai,
        "get_action_from_index",
        lambda idx, game, player_id: ("call", None),
    )
    monkeypatch.setattr(play_vs_ai.torch, "multinomial", lambda policy, k: torch.tensor([0]))

    action, amount = strategy.choose_action(game=None, player_index=0)

    assert action == "call"
    assert amount is None
    assert captured["max_seq_len"] == trainer.max_seq_len
    assert captured["feature_dim"] == trainer.history_feature_dim
