from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HUANXIN_REMOTE_ROOT = Path(
    os.environ.get("POKER_AI_HUANXIN_REMOTE_ROOT", "/root/root/work/texas-holdem")
)
DEFAULT_CHECKPOINT_DIRNAME = os.environ.get("POKER_AI_CHECKPOINT_DIRNAME", "models")
DEFAULT_TRAINED_MODELS_DIRNAME = os.environ.get(
    "POKER_AI_TRAINED_MODELS_DIRNAME", "trained_models"
)


def remote_root() -> Path:
    return DEFAULT_HUANXIN_REMOTE_ROOT


def remote_checkpoint_dir() -> Path:
    return remote_root() / DEFAULT_CHECKPOINT_DIRNAME


def remote_trained_models_dir() -> Path:
    return remote_root() / DEFAULT_TRAINED_MODELS_DIRNAME


def remote_default_model_path() -> Path:
    return remote_trained_models_dir() / "cfr_model.pth"


def remote_algorithm_checkpoint_path(algorithm: str, suffix: str) -> Path:
    safe_algorithm = algorithm.strip() or "model"
    safe_suffix = suffix.strip() or "latest"
    return remote_checkpoint_dir() / f"{safe_algorithm}_{safe_suffix}.pth"


def local_model_writes_allowed() -> bool:
    if os.environ.get("POKER_AI_ALLOW_LOCAL_MODEL_WRITES") == "1":
        return True
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    return False


def is_remote_model_path(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return False
    try:
        candidate.relative_to(remote_root())
    except ValueError:
        return False
    return True


def resolve_model_write_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if local_model_writes_allowed():
        return candidate
    if not candidate.is_absolute():
        raise ValueError(
            "Local model/checkpoint writes are disabled. "
            f"Use an absolute path under {remote_root()} on Huanxin ai1."
        )
    if not is_remote_model_path(candidate):
        raise ValueError(
            "Local model/checkpoint writes are disabled. "
            f"Use a path under {remote_root()} on Huanxin ai1."
        )
    return candidate


def prepare_model_write_path(path: str | os.PathLike[str]) -> Path:
    candidate = resolve_model_write_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate
