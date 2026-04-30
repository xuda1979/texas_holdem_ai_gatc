import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seed for random, numpy, and torch when the backend supports it."""
    random.seed(seed)
    np.random.seed(seed)

    if sys.modules.get("torch") is not torch:
        sys.modules["torch"] = torch

    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        try:
            manual_seed(seed)
        except AttributeError:
            # Some unit tests replace ``sys.modules['torch']`` or ``torch.cuda`` with
            # lightweight stubs. Keep seeding best-effort instead of crashing the suite.
            pass

    cuda = getattr(torch, "cuda", None)
    cuda_is_available = getattr(cuda, "is_available", None)
    if callable(cuda_is_available) and cuda_is_available():
        manual_seed_all = getattr(cuda, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            if hasattr(cudnn, "deterministic"):
                cudnn.deterministic = True
            if hasattr(cudnn, "benchmark"):
                cudnn.benchmark = False
