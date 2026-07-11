"""Command-line interface for comprehensive model evaluation.

Usage
-----

Evaluate a single checkpoint::

    python -m poker_ai.cli.evaluate checkpoint --path path/to/model.pth \\
        --h2h-hands 200 --health-states 32 --output report.json

Sweep every checkpoint in a directory::

    python -m poker_ai.cli.evaluate sweep --dir path/to/models \\
        --prefix deep_cfr_ --output sweep.json --latest-only

Compare two checkpoints head-to-head::

    python -m poker_ai.cli.evaluate h2h --model-a path/a.pth --model-b path/b.pth \\
        --hands 500 --output h2h.json

Run the health check only (no gameplay; fastest):

    python -m poker_ai.cli.evaluate health --path path/to/model.pth

The ``--output`` argument is optional; without it, a human-readable summary
is printed to stdout.  With it, a machine-readable JSON report is written.

All commands respect the ``POKER_AI_ALLOW_LOCAL_MODEL_WRITES=1`` env var
that the rest of the codebase uses to permit local filesystem access during
smoke tests and CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure the src/ directory is importable when this module is run as a script.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from poker_ai.evaluation.comprehensive import (  # noqa: E402
    ComprehensiveEvaluator,
    EvaluationReport,
    RegressionTracker,
    discover_checkpoints,
    load_model_from_checkpoint,
)
from poker_ai.evaluation.head_to_head import (  # noqa: E402
    AlwaysFoldStrategy,
    CallingStationStrategy,
    MatchResult,
    play_match,
)

logger = logging.getLogger("poker_ai.cli.evaluate")


def _json_default(obj: Any) -> Any:
    """JSON fallback for torch tensors / Path / devices in eval reports."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.numel() else []
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Run the full evaluation battery against a single checkpoint."""

    evaluator = ComprehensiveEvaluator(
        h2h_hands=args.h2h_hands,
        health_num_states=args.health_states,
        seed=args.seed,
        device=args.device,
        run_kuhn=args.run_kuhn,
    )
    report = evaluator.evaluate_checkpoint(args.path, total_samples=None)
    _emit_report(report, args.output)
    return 0 if report.severity != "error" else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """Evaluate every checkpoint in a directory."""

    checkpoints = discover_checkpoints(args.dir, prefix=args.prefix)
    if args.latest_only and checkpoints:
        checkpoints = checkpoints[-1:]
    if not checkpoints:
        print(f"No checkpoints found in {args.dir}"
              + (f" with prefix {args.prefix!r}" if args.prefix else ""))
        return 2

    print(f"Evaluating {len(checkpoints)} checkpoint(s) in {args.dir}")
    evaluator = ComprehensiveEvaluator(
        h2h_hands=args.h2h_hands,
        health_num_states=args.health_states,
        seed=args.seed,
        device=args.device,
        run_kuhn=args.run_kuhn,
    )
    tracker: RegressionTracker | None = None
    if args.tracker_path:
        tracker = RegressionTracker(history_path=args.tracker_path, margin_bb=args.regression_margin)

    reports: list[EvaluationReport] = []
    overall_severity = "ok"
    order = {"ok": 0, "warning": 1, "error": 2}
    for idx, ckpt in enumerate(checkpoints, 1):
        print(f"[{idx}/{len(checkpoints)}] {ckpt.name}")
        report = evaluator.evaluate_checkpoint(str(ckpt))
        if tracker is not None:
            warnings = tracker.update(report)
            for w in warnings:
                report.update_severity("warning", w)
        reports.append(report)
        print("  " + report.summary.replace("\n", "\n  "))
        if order.get(report.severity, 0) > order.get(overall_severity, 0):
            overall_severity = report.severity

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump([r.to_dict() for r in reports], fh, indent=2, default=_json_default)
        print(f"\nWrote {len(reports)} report(s) to {out}")
    print(f"\nOverall severity: {overall_severity}")
    return 0 if overall_severity != "error" else 1


def cmd_health(args: argparse.Namespace) -> int:
    """Run only the (cheap) health check, no gameplay."""

    try:
        model, metadata, device = load_model_from_checkpoint(args.path, device=args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: could not load checkpoint: {exc}")
        return 1
    from poker_ai.evaluation.health import run_health_check

    num_actions = int(metadata.get("num_actions") or 10)
    max_seq_len = int(metadata.get("max_seq_len") or 256)
    d_raw = int(metadata.get("history_feature_dim") or metadata.get("d_raw_feature") or 18)
    report = run_health_check(
        model,
        checkpoint_path=args.path,
        metadata=metadata,
        num_states=args.health_states,
        num_actions=num_actions,
        max_seq_len=max_seq_len,
        d_raw_feature=d_raw,
        seed=args.seed,
        device=device,
    )
    print(report.summary())
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(
                {
                    "checkpoint_path": args.path,
                    "severity": report.severity,
                    "ok": report.ok,
                    "checks": report.checks,
                    "diagnostics": report.diagnostics,
                },
                fh,
                indent=2,
            )
    return 0 if report.severity != "error" else 1


def cmd_h2h(args: argparse.Namespace) -> int:
    """Play a head-to-head match between two checkpoints (A vs B)."""

    try:
        model_a, meta_a, device_a = load_model_from_checkpoint(args.model_a, device=args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: could not load model A: {exc}")
        return 1
    try:
        model_b, meta_b, device_b = load_model_from_checkpoint(args.model_b, device=args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: could not load model B: {exc}")
        return 1

    from poker_ai.gui.playStrategy import ModelAIStrategy

    def _config(meta: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(meta)
        cfg.setdefault("input_feature_dim", meta.get("history_feature_dim") or 18)
        cfg.setdefault("d_raw_feature", meta.get("history_feature_dim") or 18)
        cfg.setdefault("max_seq_len", meta.get("max_seq_len") or 256)
        return cfg

    strat_a = ModelAIStrategy(model_a, _config(meta_a), device_a)
    strat_b = ModelAIStrategy(model_b, _config(meta_b), device_b)

    result = play_match(
        strat_a,
        strat_b,
        num_hands=args.hands,
        starting_stack=args.starting_stack,
        big_blind=args.big_blind,
        small_blind=args.small_blind,
        seed=args.seed,
    )
    _print_match_result("A", args.model_a, "B", args.model_b, result)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(
                {
                    "model_a": args.model_a,
                    "model_b": args.model_b,
                    "result": {
                        "hands_played": result.hands_played,
                        "bb_per_100": result.bb_per_100,
                        "stderr_bb_per_100": result.stderr_bb_per_100,
                        "total_chips": result.total_chips,
                        "big_blind": result.big_blind,
                    },
                },
                fh,
                indent=2,
            )
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Play the model at ``--path`` against the full baseline panel."""

    evaluator = ComprehensiveEvaluator(
        h2h_hands=args.hands,
        health_num_states=0,  # skip health
        seed=args.seed,
        device=args.device,
        run_kuhn=False,
    )
    # Build a no-health evaluator by overriding baselines if requested.
    if args.no_health:
        evaluator.health_num_states = 0
    report = evaluator.evaluate_checkpoint(args.path)
    for h2h in report.h2h_results:
        print(
            f"  {h2h['baseline']:>16}: bb/100={h2h['bb_per_100']:>8.2f} "
            f"± {h2h['stderr_bb_per_100']:>7.2f}  ({h2h['hands_played']} hands)"
        )
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            fh.write(report.to_json())
    return 0 if report.severity != "error" else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit_report(report: EvaluationReport, output_path: str | None) -> None:
    print(report.summary)
    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            fh.write(report.to_json())
        print(f"\nWrote report to {out}")


def _print_match_result(
    a_name: str, a_path: str, b_name: str, b_path: str, result: MatchResult
) -> None:
    print(f"Match: {a_name} ({a_path}) vs {b_name} ({b_path})")
    print(f"  hands played:     {result.hands_played}")
    print(f"  bb/100 (A's POV): {result.bb_per_100:.2f} ± {result.stderr_bb_per_100:.2f}")
    print(f"  total chips (A):  {result.total_chips:.2f}")
    print(f"  big blind:        {result.big_blind}")
    if result.is_significantly_positive():
        print(f"  verdict:          {a_name} significantly wins")
    elif result.is_significantly_negative():
        print(f"  verdict:          {b_name} significantly wins")
    else:
        print("  verdict:          no significant difference")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poker_ai evaluate",
        description="Comprehensive evaluation of poker AI checkpoints.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # checkpoint -----------------------------------------------------------
    p_ckpt = sub.add_parser("checkpoint", help="Evaluate a single checkpoint")
    p_ckpt.add_argument("--path", required=True, help="Path to .pth checkpoint")
    p_ckpt.add_argument("--h2h-hands", type=int, default=200)
    p_ckpt.add_argument("--health-states", type=int, default=32)
    p_ckpt.add_argument("--seed", type=int, default=7)
    p_ckpt.add_argument("--device", default="cpu")
    p_ckpt.add_argument("--run-kuhn", action="store_true")
    p_ckpt.add_argument("--output", help="Write JSON report to this path")
    p_ckpt.set_defaults(func=cmd_checkpoint)

    # sweep ----------------------------------------------------------------
    p_sweep = sub.add_parser("sweep", help="Evaluate every checkpoint in a directory")
    p_sweep.add_argument("--dir", required=True, help="Directory containing .pth files")
    p_sweep.add_argument("--prefix", default=None, help="Filter by filename prefix")
    p_sweep.add_argument("--latest-only", action="store_true", help="Only evaluate the newest checkpoint")
    p_sweep.add_argument("--h2h-hands", type=int, default=100)
    p_sweep.add_argument("--health-states", type=int, default=16)
    p_sweep.add_argument("--seed", type=int, default=7)
    p_sweep.add_argument("--device", default="cpu")
    p_sweep.add_argument("--run-kuhn", action="store_true")
    p_sweep.add_argument("--output", help="Write combined JSON report to this path")
    p_sweep.add_argument("--tracker-path", help="Optional regression tracker history file")
    p_sweep.add_argument("--regression-margin", type=float, default=10.0, help="bb/100 margin for regression detection")
    p_sweep.set_defaults(func=cmd_sweep)

    # health ---------------------------------------------------------------
    p_health = sub.add_parser("health", help="Run only the (fast) health check")
    p_health.add_argument("--path", required=True, help="Path to .pth checkpoint")
    p_health.add_argument("--health-states", type=int, default=32)
    p_health.add_argument("--seed", type=int, default=7)
    p_health.add_argument("--device", default="cpu")
    p_health.add_argument("--output", help="Write JSON report to this path")
    p_health.set_defaults(func=cmd_health)

    # h2h ------------------------------------------------------------------
    p_h2h = sub.add_parser("h2h", help="Play two checkpoints head-to-head")
    p_h2h.add_argument("--model-a", required=True, help="Path to model A .pth")
    p_h2h.add_argument("--model-b", required=True, help="Path to model B .pth")
    p_h2h.add_argument("--hands", type=int, default=200)
    p_h2h.add_argument("--starting-stack", type=int, default=1000)
    p_h2h.add_argument("--big-blind", type=int, default=10)
    p_h2h.add_argument("--small-blind", type=int, default=5)
    p_h2h.add_argument("--seed", type=int, default=7)
    p_h2h.add_argument("--device", default="cpu")
    p_h2h.add_argument("--output", help="Write JSON report to this path")
    p_h2h.set_defaults(func=cmd_h2h)

    # baselines ------------------------------------------------------------
    p_base = sub.add_parser("baselines", help="Play model against the full baseline panel")
    p_base.add_argument("--path", required=True, help="Path to .pth checkpoint")
    p_base.add_argument("--hands", type=int, default=200)
    p_base.add_argument("--no-health", action="store_true", help="Skip health checks")
    p_base.add_argument("--seed", type=int, default=7)
    p_base.add_argument("--device", default="cpu")
    p_base.add_argument("--output", help="Write JSON report to this path")
    p_base.set_defaults(func=cmd_baselines)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("POKER_AI_EVAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
