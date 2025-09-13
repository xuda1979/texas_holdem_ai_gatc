"""AI utilities and model loading.

The heavy model-loading logic is imported lazily to avoid circular imports with
GUI components.  ``load_model_strategy`` will only pull in the actual loader
when called.
"""

__all__ = ["load_model_strategy"]


def load_model_strategy(*args: object, **kwargs: object) -> object:
    """Return an AI strategy loader with lazy imports."""  # pragma: no cover - thin wrapper
    from .model_loader import load_model_strategy as _load

    return _load(*args, **kwargs)
