from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
ENV_CONFIG_PATH = "POKER_AI_CONFIG"
ENV_PREFIX = "POKER_AI__"


_FALLBACK_CONFIG: dict[str, Any] = {
    "game_engine": {
        "num_players": 2,
        "starting_stack": 10000,
        "small_blind": 10,
        "big_blind": 20,
    },
    "training": {
        "num_training_hands": 2_000_000,
        "save_model_every_n_hands": 10_000,
        "save_model_every_minutes": 10,
        "log_every_n_hands": 1000,
        "cfr_algorithm": "vanilla",
        "cfr_discount_factor": 1.0,
        "distributed_workers": 1,
        "save_model_path": "trained_models/cfr_model.pth",
        "min_buffer_before_train": 256,
    },
    "model": {
        "directory": "trained_models/",
        "filename_prefix": "cfr_model",
        "hidden_dim": 768,
        "num_actions": 10,
        "d_raw_feature": 18,
        "num_layers": 12,
        "num_heads": 12,
    },
    "self_play": {
        "save_interval": 1800,
        "temperature": 1.0,
        "epsilon": 0.05,
    },
    "player_strategies": ["cfr_trained", "random"],
    "simulation": {
        "num_simulation_hands": 1000,
        "log_results_to_file": True,
        "results_filename": "simulation_results.txt",
    },
    "abstraction": {"enabled": False, "buckets": 5},
    "api_server": {"host": "0.0.0.0", "port": 5001, "debug_mode": True},
    "evaluation": {"best_response_iterations": 100},
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_to_file": True,
        "log_file": "poker_ai.log",
    },
}


_YAML: ModuleType | None = None
_YAML_IMPORT_ATTEMPTED = False


def _import_yaml() -> ModuleType | None:
    global _YAML_IMPORT_ATTEMPTED, _YAML
    if not _YAML_IMPORT_ATTEMPTED:
        try:
            import yaml
        except ImportError:  # pragma: no cover - handled in loader
            print(
                "Warning: PyYAML is not installed. Configuration loading may be limited."
            )
            _YAML = None
        else:
            _YAML = yaml
        finally:
            _YAML_IMPORT_ATTEMPTED = True
    return _YAML


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    yaml = _import_yaml()
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        keys = key[len(ENV_PREFIX) :].lower().split("__")
        target = cfg
        for part in keys[:-1]:
            target = target.setdefault(part, {})
        if yaml is None:
            target[keys[-1]] = value
        else:
            try:
                target[keys[-1]] = yaml.safe_load(value)
            except yaml.YAMLError:
                target[keys[-1]] = value


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load configuration from YAML and apply environment overrides.

    Parameters
    ----------
    path:
        Optional path to a configuration file.  If ``None`` the loader will look
        for ``POKER_AI_CONFIG`` environment variable and fall back to the default
        ``config.yaml`` bundled with the package.
    """
    yaml = _import_yaml()
    path_str = path or os.environ.get(ENV_CONFIG_PATH)
    config_path = Path(path_str) if path_str else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if yaml is None:
        resolved_default = DEFAULT_CONFIG_PATH.resolve()
        resolved_requested = config_path.resolve()
        if path_str and resolved_requested != resolved_default:
            raise RuntimeError(
                "PyYAML is required to load configuration files "
                f"(attempted path: {config_path}). "
                "Install the 'PyYAML' package to enable configuration loading."
            )
        data = deepcopy(_FALLBACK_CONFIG)
        _apply_env_overrides(data)
        return data
    try:
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default configurations.")
        data = {}
    except yaml.YAMLError as e:
        print(f"Error parsing {config_path}: {e}. Using default configurations.")
        data = {}
    _apply_env_overrides(data)
    return data
