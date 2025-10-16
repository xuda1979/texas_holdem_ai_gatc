"""Smoke tests for the subsystem architecture."""

from __future__ import annotations

import torch

from poker_ai.systems import (
    CFRSubsystem,
    EmbeddingSubsystem,
    EvaluationSubsystem,
    LoggingSubsystem,
    RulesSubsystem,
    SelfPlaySubsystem,
    TrainingSubsystem,
    TransformerSubsystem,
)


def test_subsystem_construction_and_registry():
    cfr = CFRSubsystem.create(variant="ai", device="cpu")
    rules = RulesSubsystem.create(num_players=2, starting_stack=50, verbose=False)
    transformer = TransformerSubsystem.create(
        history_feature_dim=cfr.component.history_feature_dim,
        card_feature_dim=cfr.component.card_feature_dim,
        hidden_dim=cfr.component.hidden_dim,
        num_heads=cfr.component.num_heads,
        num_layers=cfr.component.num_layers,
        num_actions=cfr.component.num_actions,
    )
    embeddings = EmbeddingSubsystem.create(
        history_feature_dim=cfr.component.history_feature_dim,
        card_feature_dim=cfr.component.card_feature_dim,
        max_seq_len=cfr.component.max_seq_len,
    )

    assert transformer.component.num_actions == cfr.component.num_actions
    transformer.freeze()

    logging_system = LoggingSubsystem.create()
    evaluation = EvaluationSubsystem.create()

    self_play = SelfPlaySubsystem.create(
        cfr_trainer=cfr.component,
        game_engine_config={
            "starting_stack": 20,
            "big_blind": 1,
            "small_blind": 1,
            "min_players": 2,
            "max_players": 2,
        },
        training_config={"train_during_generation": False},
        train_during_generation=False,
    )

    training = TrainingSubsystem.create(cfr=cfr, self_play=self_play, iterations_per_cycle=0)

    # Exercise the embedding pipeline on a fresh game.
    game = rules.new_game(num_players=2, starting_stack=20, verbose=False)
    game.initialize_game()
    hole, community, history, mask = embeddings.component.build_inputs(game, player_index=0, return_mask=True)

    assert hole.shape[-1] == embeddings.component.card_feature_dim
    assert community.shape[-1] == embeddings.component.card_feature_dim
    assert history.shape[-1] == embeddings.component.history_feature_dim
    assert mask is not None

    # Ensure tensors can be moved to the trainer's device.
    device = torch.device("cpu")
    hole_dev, community_dev, history_dev, mask_dev = embeddings.component.build_inputs(
        game,
        player_index=0,
        return_mask=True,
        device=device,
    )
    assert hole_dev.device == device
    assert community_dev.device == device
    assert history_dev.device == device
    assert mask_dev is not None and mask_dev.device == device

    # Make sure the evaluation subsystem exposes the expected callable.
    assert hasattr(evaluation.component, "run_tournament")

    # The training subsystem with zero iterations should not trigger self-play.
    training.component.run_cycle(start_iteration=0)
