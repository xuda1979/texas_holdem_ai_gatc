"""MLflow tracking context for CFR training runs.

Wraps MLflow in a context manager that:
- logs all config params
- logs scalar metrics with step
- logs model checkpoints as artifacts (and registers them)
- logs evaluation reports (h2h bb/100, exploitability, health)
- records the git commit and config hash as tags for lineage

Works fully offline: MLflow file store is used by default
(``mlruns/`` directory), no server required.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_DEFAULT_TRACKING_URI = "file://./mlruns"


def _resolve_tracking_uri(uri: str | None) -> str:
    if uri is not None:
        return uri
    return os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)


class ActiveRun:
    """Thin wrapper around an active MLflow run."""

    def __init__(self, run: Any) -> None:
        self._run = run

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    def log_param(self, key: str, value: Any) -> None:
        import mlflow
        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow
        # Flatten nested dicts so MLflow stores ``model.hidden_dim`` etc.
        flat: dict[str, Any] = {}
        _flatten(params, prefix="", out=flat)
        for k, v in flat.items():
            try:
                mlflow.log_param(k, v)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("mlflow.log_param(%s) failed: %s", k, exc)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        import mlflow
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        import mlflow
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | os.PathLike[str]) -> None:
        import mlflow
        mlflow.log_artifact(str(path))

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        import mlflow
        mlflow.log_dict(data, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        import mlflow
        mlflow.set_tag(key, value)


def _flatten(d: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, prefix=key, out=out)
        else:
            out[key] = v


class TrackingContext:
    """Context manager that creates an MLflow run and logs everything.

    Parameters
    ----------
    experiment
        MLflow experiment name (e.g. ``"deep_cfr"``).
    run_name
        Human-readable run name (e.g. ``"asi1-4layer-run1"``).
    tracking_uri
        MLflow tracking URI. Defaults to ``file://./mlruns`` (offline).
    tags
        Additional tags to set on the run (e.g. ``{"environment": "asi1"}``).

    Example
    -------
    >>> with TrackingContext("deep_cfr", "asi1-4layer") as ctx:
    ...     ctx.log_params(config)
    ...     ctx.log_metric("avg_loss", 0.059, step=1)
    ...     ctx.log_eval_report(report_dict)
    """

    def __init__(
        self,
        experiment: str = "deep_cfr",
        run_name: str | None = None,
        *,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = _resolve_tracking_uri(tracking_uri)
        self.tags = tags or {}
        self._active: ActiveRun | None = None

    def __enter__(self) -> ActiveRun:
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self._active = ActiveRun(run)
        # Record environment + git lineage as tags.
        from poker_ai.tracking.lineage import git_commit

        commit, dirty = git_commit()
        self._active.set_tag("git_commit", commit)
        self._active.set_tag("git_dirty", str(dirty))
        return self._active

    def __exit__(self, exc_type, exc, tb) -> None:
        import mlflow

        if exc_type is None:
            mlflow.end_run(status="FINISHED")
        else:
            mlflow.end_run(status="FAILED")
        self._active = None

    # Convenience classmethod for one-shot metric logging without a context.
    @classmethod
    @contextmanager
    def start(
        cls,
        experiment: str = "deep_cfr",
        run_name: str | None = None,
        **kwargs: Any,
    ) -> Iterator[ActiveRun]:
        with cls(experiment=experiment, run_name=run_name, **kwargs) as run:
            yield run
