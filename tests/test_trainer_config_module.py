from __future__ import annotations

import argparse
import sys
import types

from poker_ai.ai import trainers as trainer_module
from poker_ai.ai.trainers.ai_cfr_trainer import _publish_config_module
from training.train import _override_config


def test_trainer_config_is_published_as_module() -> None:
    config_module = sys.modules["poker_ai.ai.trainers.config"]
    assert isinstance(config_module, types.ModuleType)
    assert config_module.config is trainer_module.config
    assert config_module.model == trainer_module.config["model"]
    assert config_module.training == trainer_module.config["training"]


def test_override_config_keeps_module_entry_hashable_and_synced() -> None:
    args = argparse.Namespace(
        lightweight=True,
        hidden_dim=64,
        num_layers=None,
        num_heads=None,
        learning_rate=None,
        max_seq_len=None,
        replay_buffer_capacity=None,
        save_path=None,
        log_file=None,
        log_level=None,
    )

    original_config = trainer_module.config
    try:
        new_config = _override_config(args)
        config_module = sys.modules["poker_ai.ai.trainers.config"]
        assert isinstance(config_module, types.ModuleType)
        assert hash(config_module)
        assert config_module.config is new_config
        assert config_module.model is new_config["model"]
        assert new_config["model"]["hidden_dim"] == 64
    finally:
        trainer_module.config = original_config
        trainer_module.ai_cfr_trainer_module.config = original_config
        _publish_config_module(original_config)
