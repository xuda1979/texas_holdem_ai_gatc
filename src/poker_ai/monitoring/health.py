import math
from typing import Any, Iterable, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


def _is_nan(x: float) -> bool:
    return math.isnan(x)


def _is_inf(x: float) -> bool:
    return math.isinf(x)


def has_nan_or_inf(x: Any) -> bool:
    """
    Return True if x contains a NaN or Inf. Works with floats, lists, tuples,
    numpy arrays, and torch tensors (if torch is available).
    """
    if isinstance(x, (float, int)):
        return _is_nan(float(x)) or _is_inf(float(x))

    # Torch tensor
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return bool(torch.isnan(x).any() or torch.isinf(x).any())

    # Numpy-like / iterable fallback
    if hasattr(x, "__iter__"):
        for v in x:
            if has_nan_or_inf(v):
                return True
        return False

    return False


def check_probability_vector(vec: Iterable[float], tol: float = 1e-5) -> Optional[str]:
    """
    Checks that probabilities are within [0,1] and sum to 1 (within tolerance).
    Returns an error string if invalid, otherwise None.
    """
    s = 0.0
    for v in vec:
        if v < -tol or v > 1.0 + tol or _is_nan(v) or _is_inf(v):
            return f"probability out of range or NaN/Inf: {v}"
        s += v
    if abs(s - 1.0) > max(tol, 1e-8):
        return f"probabilities do not sum to 1 (sum={s})"
    return None


def clip_grad_norm_if_available(model: Any, max_norm: float) -> Optional[float]:
    """
    If torch is available, clips gradient norm for model parameters and returns the norm.
    Otherwise returns None (no-op).
    """
    if not _HAS_TORCH:
        return None
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
    return norm

