"""Utility functions for reliable training loops.

These helpers focus on deterministic behaviour and checkpoint
management so tests can reason about training without touching the
rest of the project.  The functions are intentionally lightweight and
only depend on PyTorch.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import random
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on environment
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> float:
    """Run a single optimisation step using an entire batch."""
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return float(loss.item())


def train_sequential(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> float:
    """Train on each sample sequentially but accumulate gradients."""
    model.train()
    optimizer.zero_grad()
    n = inputs.size(0)
    total_loss = 0.0
    for x, y in zip(inputs, targets, strict=False):
        out = model(x.unsqueeze(0))
        loss = loss_fn(out, y.unsqueeze(0)) / n
        loss.backward()
        total_loss += float(loss.item())
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return total_loss


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def _clone_grads(model: torch.nn.Module) -> List[torch.Tensor]:
    """Return a list of detached gradient tensors for ``model``."""
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def batch_gradients(
    model: torch.nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[List[torch.Tensor], float]:
    """Compute gradients for a full batch without updating parameters."""
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    return _clone_grads(model), float(loss.item())


def sequential_gradients(
    model: torch.nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[List[torch.Tensor], float]:
    """Compute gradients by processing samples one at a time."""
    model.zero_grad(set_to_none=True)
    n = inputs.size(0)
    total_loss = 0.0
    for x, y in zip(inputs, targets, strict=False):
        out = model(x.unsqueeze(0))
        loss = loss_fn(out, y.unsqueeze(0)) / n
        loss.backward()
        total_loss += float(loss.item())
    return _clone_grads(model), total_loss


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    **extra: Any,
) -> None:
    """Persist training state to ``path``."""
    ckpt: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    ckpt.update(extra)
    torch.save(ckpt, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> Dict[str, Any]:
    """Load state from ``path`` and restore model/optimizer/scheduler."""
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        k: v
        for k, v in ckpt.items()
        if k not in {"model", "optimizer", "scheduler"}
    }
