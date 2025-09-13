from __future__ import annotations

from pathlib import Path

import yaml

from poker_ai.config import load_config


CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "poker_ai" / "config" / "config.yaml"


def test_config_loader_matches_yaml() -> None:
    """Ensure the loader returns the exact contents of config.yaml."""
    loaded = load_config(CONFIG_PATH)
    with CONFIG_PATH.open() as fh:
        expected = yaml.safe_load(fh)
    assert loaded == expected


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("POKER_AI__GAME_ENGINE__NUM_PLAYERS", "5")
    cfg = load_config(CONFIG_PATH)
    assert cfg["game_engine"]["num_players"] == 5
