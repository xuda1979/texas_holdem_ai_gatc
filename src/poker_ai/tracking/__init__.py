"""Experiment tracking for CFR poker AI.

Integrates:
- MLflow for experiment/run tracking (params, metrics, artifacts, model registry)
- DVC for data/model versioning (git-like semantics for .pth checkpoints)
- Git for code lineage (commit hash recorded with every run)

Usage in training::

    from poker_ai.tracking import TrackingContext
    with TrackingContext(experiment="deep_cfr", run_name="asi1-4layer") as ctx:
        ctx.log_params(config_dict)
        ctx.log_metric("avg_loss", 0.059, step=cycle)
        ctx.log_model(checkpoint_path, model_name="deep_cfr_advantage")
        ctx.log_eval_report(eval_report_dict)
"""
from poker_ai.tracking.context import TrackingContext, ActiveRun
from poker_ai.tracking.lineage import (
    capture_lineage,
    LineageRecord,
    write_lineage_manifest,
    read_lineage_manifest,
)

__all__ = [
    "TrackingContext",
    "ActiveRun",
    "capture_lineage",
    "LineageRecord",
    "write_lineage_manifest",
    "read_lineage_manifest",
]
