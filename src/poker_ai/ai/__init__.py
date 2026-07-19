"""AI utilities and model loading.

The heavy model-loading logic is imported lazily to avoid circular imports with
GUI components.  ``load_model_strategy`` will only pull in the actual loader
when called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import only for static type checkers; avoid circular import at runtime.
    from poker_ai.gui.playStrategy import PlayerStrategy

__all__ = ["load_model_strategy"]


def load_model_strategy(
    model_path: str = "models/best/best.pth",
) -> "tuple[PlayerStrategy, Any]":
    """Return an AI strategy loader with lazy imports."""  # pragma: no cover - thin wrapper
    from .model_loader import DEFAULT_MODEL_PATH, load_model_strategy as _load

    if model_path == "models/best/best.pth":
        return _load(DEFAULT_MODEL_PATH)
    return _load(model_path)

