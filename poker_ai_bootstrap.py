"""
Bootstrap for running repo entrypoints directly from a checkout.
  - Ensures <repo>/src is at the front of sys.path (avoid import shadowing by top-level 'poker_ai').
  - Provides a single seed_all() for deterministic runs (Python, NumPy, PyTorch if available).
Usage:
    import poker_ai_bootstrap as _pab
    _pab.seed_all()  # or set RUN_DETERMINISTIC=1 in the environment
"""
from __future__ import annotations

import os
import sys
import random
from pathlib import Path

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may not be installed yet
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is optional in some setups
    torch = None  # type: ignore


# --- 1) Put <repo>/src at the front of sys.path -----------------------------
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir():
    src_str = str(_SRC)
    if not any(p == src_str for p in sys.path[:3]):  # keep it very early
        sys.path.insert(0, src_str)
del _ROOT, _SRC


# --- 2) Deterministic seeding -----------------------------------------------
def seed_all(seed: int | None = None) -> int:
    """
    Seed Python, NumPy, and PyTorch RNGs. If seed is None, read from $SEED or 0.
    Returns the actual seed used.
    """
    if seed is None:
        raw = os.environ.get("SEED", "").strip()
        seed = int(raw) if raw.isdigit() else 0

    try:
        random.seed(seed)
    except Exception:
        pass

    if np is not None:
        try:
            # numpy expects uint32 range
            np.random.seed(seed % (2**32 - 1))
        except Exception:
            pass

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # For extra determinism when desired; may reduce performance.
            if os.environ.get("TORCH_DETERMINISTIC", "").lower() in {"1", "true", "yes"}:
                torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    return seed


# Auto-seed when requested by environment
if os.environ.get("RUN_DETERMINISTIC", "").lower() in {"1", "true", "yes"}:
    seed_all()
