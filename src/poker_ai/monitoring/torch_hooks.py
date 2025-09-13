from typing import Any, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

from .logger import MetricsLogger
from .health import has_nan_or_inf


def attach_gradient_health_hooks(model: Any, logger: Optional[MetricsLogger] = None) -> None:
    """
    If PyTorch is available, attach per-parameter gradient hooks to detect NaN/Inf
    gradients and optionally log grad norms. This is a no-op if torch is not installed.
    """
    if not _HAS_TORCH:
        return

    def _hook(param_name: str):
        def _cb(grad):
            if grad is None:
                return
            if has_nan_or_inf(grad):
                if logger:
                    logger.log("grad_nan_or_inf", 1.0, extra={"param": param_name})
                raise FloatingPointError(f"NaN/Inf gradient detected at param: {param_name}")
        return _cb

    for name, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(_hook(name))

