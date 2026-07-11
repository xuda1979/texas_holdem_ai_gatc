"""Lineage tracking: record exactly what produced each model checkpoint.

A :class:`LineageRecord` captures:
- code_commit: the git commit hash at training time
- config_hash: sha256 of the training config YAML
- data_hash: sha256 of the replay buffer / training data (if available)
- model_path: path to the .pth checkpoint
- model_hash: sha256 of the checkpoint file
- metrics: dict of evaluation metrics at checkpoint time
- created_at: UTC timestamp

Lineage manifests are written as JSON to ``models/tracked/<run_id>/lineage.json``
and can be queried later to answer "what produced this model?".
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LineageRecord:
    """Full provenance record for a single model checkpoint."""

    run_id: str
    model_path: str
    model_hash: str
    code_commit: str
    code_commit_dirty: bool
    config_hash: str
    config_path: str
    data_hash: str | None = None
    data_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    algorithm: str | None = None
    environment: str | None = None
    npus_used: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)


def git_commit(repo_dir: str | os.PathLike[str] = ".") -> tuple[str, bool]:
    """Return ``(commit_hash, is_dirty)`` for ``repo_dir``.

    Falls back to ``("unknown", True)`` if git is unavailable or the dir is
    not a repository.
    """
    repo = Path(repo_dir)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown", True


def file_hash(path: str | os.PathLike[str], algo: str = "sha256") -> str:
    """Return the hex digest of ``path`` using ``algo``."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict[str, Any] | str | os.PathLike[str]) -> str:
    """Return a stable sha256 of a config dict or YAML file path."""
    if isinstance(config, dict):
        payload = json.dumps(config, sort_keys=True, default=str).encode()
    else:
        payload = Path(config).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def capture_lineage(
    run_id: str,
    model_path: str | os.PathLike[str],
    config: dict[str, Any] | str | os.PathLike[str],
    *,
    repo_dir: str | os.PathLike[str] = ".",
    data_path: str | os.PathLike[str] | None = None,
    metrics: dict[str, Any] | None = None,
    algorithm: str | None = None,
    environment: str | None = None,
    npus_used: int | None = None,
) -> LineageRecord:
    """Capture the full lineage of ``model_path`` in one call.

    Parameters
    ----------
    run_id
        The training run identifier (e.g. ``"train-20260706T103243"``).
    model_path
        Path to the ``.pth`` checkpoint file.
    config
        Either the config dict or the path to the YAML config file.
    repo_dir
        Repository root for git commit lookup.
    data_path
        Optional path to the training data / replay buffer file.
    metrics
        Optional dict of evaluation metrics to record with the checkpoint.
    algorithm
        Training algorithm name (e.g. ``"deep_cfr"``).
    environment
        Training environment name (e.g. ``"asi1"``).
    npus_used
        Number of NPUs used for training.
    """
    commit, dirty = git_commit(repo_dir)
    return LineageRecord(
        run_id=run_id,
        model_path=str(model_path),
        model_hash=file_hash(model_path),
        code_commit=commit,
        code_commit_dirty=dirty,
        config_hash=config_hash(config),
        config_path=str(config) if not isinstance(config, dict) else "",
        data_hash=file_hash(data_path) if data_path and Path(data_path).exists() else None,
        data_path=str(data_path) if data_path else None,
        metrics=metrics or {},
        algorithm=algorithm,
        environment=environment,
        npus_used=npus_used,
    )


def write_lineage_manifest(
    record: LineageRecord,
    dest_dir: str | os.PathLike[str] = "models/tracked",
) -> Path:
    """Write ``record`` as ``<dest_dir>/<run_id>/lineage.json``.

    Returns the path to the written manifest.
    """
    out = Path(dest_dir) / record.run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "lineage.json"
    manifest.write_text(record.to_json(), encoding="utf-8")
    return manifest


def read_lineage_manifest(
    run_id: str,
    dest_dir: str | os.PathLike[str] = "models/tracked",
) -> LineageRecord:
    """Read a lineage manifest written by :func:`write_lineage_manifest`."""
    manifest = Path(dest_dir) / run_id / "lineage.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return LineageRecord(**data)
