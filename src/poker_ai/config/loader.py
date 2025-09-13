import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
ENV_CONFIG_PATH = "POKER_AI_CONFIG"
ENV_PREFIX = "POKER_AI__"


def _import_yaml():
    try:
        import yaml
    except ImportError:  # pragma: no cover - handled in loader
        print("Warning: PyYAML is not installed. Using default configurations.")
        yaml = None
    return yaml


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


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load configuration from YAML and apply environment overrides.

    Parameters
    ----------
    path:
        Optional path to a configuration file.  If ``None`` the loader will look
        for ``POKER_AI_CONFIG`` environment variable and fall back to the default
        ``config.yaml`` bundled with the package.
    """
    yaml = _import_yaml()
    config_path = Path(path or os.environ.get(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH))
    data: dict[str, Any] = {}
    if yaml is None:
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
