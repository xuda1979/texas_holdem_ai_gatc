"""Utilities for discovering saved model checkpoints."""

from __future__ import annotations

from pathlib import Path

from poker_ai.config import MODEL_DIR

__all__ = ["find_latest_model_checkpoint"]


def _iter_checkpoint_files(directory: Path) -> list[Path]:
    """Yield checkpoint files within ``directory`` sorted by modification time."""

    if not directory.exists():
        return []

    files = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix == ".pth"
    ]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files


def find_latest_model_checkpoint(
    directory: str | Path | None = None, prefix: str | None = None
) -> tuple[str, float] | None:
    """Return the newest ``.pth`` checkpoint path under ``directory``.

    Parameters
    ----------
    directory:
        Directory to scan for model checkpoints. Defaults to
        :data:`poker_ai.config.MODEL_DIR`.
    prefix:
        Optional filename prefix filter. If provided, only files whose names
        start with ``prefix`` are considered.

    Returns
    -------
    tuple[str, float] | None
        The path to the newest checkpoint and its modification time. ``None``
        if no suitable checkpoint exists.
    """

    root = Path(directory or MODEL_DIR)

    candidates = _iter_checkpoint_files(root)
    if prefix is not None:
        candidates = [path for path in candidates if path.name.startswith(prefix)]

    for path in candidates:
        stat = path.stat()
        return str(path), stat.st_mtime

    if prefix is not None:
        # Retry without prefix filtering when no matching file exists.
        return find_latest_model_checkpoint(directory=root, prefix=None)

    return None
