from __future__ import annotations

import time
from pathlib import Path

from poker_ai.utils.model_paths import find_latest_model_checkpoint


def test_find_latest_model_checkpoint_empty(tmp_path: Path) -> None:
    assert find_latest_model_checkpoint(directory=tmp_path) is None


def test_find_latest_model_checkpoint_returns_latest(tmp_path: Path) -> None:
    first = tmp_path / "first_model.pth"
    second = tmp_path / "second_model.pth"
    first.write_bytes(b"first")
    time.sleep(0.01)
    second.write_bytes(b"second")

    latest = find_latest_model_checkpoint(directory=tmp_path)
    assert latest is not None
    path, _ = latest
    assert path == str(second)


def test_find_latest_model_checkpoint_prefix_fallback(tmp_path: Path) -> None:
    base = tmp_path / "base_model.pth"
    base.write_bytes(b"base")

    prefixed = tmp_path / "custom_model.pth"
    time.sleep(0.01)
    prefixed.write_bytes(b"custom")

    latest_prefixed = find_latest_model_checkpoint(directory=tmp_path, prefix="custom")
    assert latest_prefixed is not None
    path, _ = latest_prefixed
    assert path == str(prefixed)

    # Asking for a missing prefix should fall back to the newest checkpoint.
    fallback = find_latest_model_checkpoint(directory=tmp_path, prefix="missing")
    assert fallback is not None
    fallback_path, _ = fallback
    assert fallback_path == str(prefixed)
