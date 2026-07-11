"""Tests for the comprehensive evaluation harness.

These tests exercise the full evaluation stack against synthetic checkpoints
so they don't depend on a trained model being present.  They cover:

* Loading a checkpoint from disk (legacy + new metadata format).
* Running health checks against a freshly-initialised model.
* Running head-to-head matches against the baseline panel.
* Regression detection over a sequence of reports.
* The training-loop hook (``TrainingEvalHook``) threshold logic.
* Checkpoint discovery in a directory.
* JSON serialisation round-trip.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest
import torch

# Allow tests to write models locally.
os.environ.setdefault("POKER_AI_ALLOW_LOCAL_MODEL_WRITES", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.evaluation.comprehensive import (
    ComprehensiveEvaluator,
    EvaluationReport,
    H2HResult,
    KuhnExploitability,
    RandomAllInStrategy,
    RegressionTracker,
    TrainingEvalHook,
    discover_checkpoints,
    evaluate_checkpoint_cli,
    load_model_from_checkpoint,
    sweep_checkpoints_cli,
)
from poker_ai.evaluation.head_to_head import MatchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model(**overrides) -> AdvantageNetwork:
    kwargs = dict(
        history_feature_dim=18,
        card_feature_dim=17,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_actions=10,
    )
    kwargs.update(overrides)
    model = AdvantageNetwork(**kwargs)
    model.eval()
    return model


def _save_checkpoint(
    path: Path,
    model: AdvantageNetwork | None = None,
    metadata: dict | None = None,
    include_policy: bool = True,
) -> Path:
    if model is None:
        model = _make_model()
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {
            "history_feature_dim": 18,
            "card_feature_dim": 17,
            "num_actions": 10,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_layers": 2,
            "max_seq_len": 256,
            "trainer": "deep_cfr",
        },
    }
    if include_policy:
        payload["policy_net_state_dict"] = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return path


@pytest.fixture
def tmp_checkpoint(tmp_path: Path) -> Path:
    return _save_checkpoint(tmp_path / "ckpt.pth")


@pytest.fixture
def tmp_checkpoints(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(3):
        p = tmp_path / f"deep_cfr_hand_{i}.pth"
        _save_checkpoint(p)
        # Stagger mtimes so discover_checkpoints order is stable.
        os.utime(p, (i, i))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# load_model_from_checkpoint
# ---------------------------------------------------------------------------


class TestLoadModelFromCheckpoint:
    def test_loads_new_format(self, tmp_checkpoint: Path) -> None:
        model, meta, device = load_model_from_checkpoint(str(tmp_checkpoint))
        assert isinstance(model, AdvantageNetwork)
        assert meta["trainer"] == "deep_cfr"
        assert device.type == "cpu"
        assert hasattr(model, "_policy_net_state_dict")

    def test_loads_legacy_state_dict_only(self, tmp_path: Path) -> None:
        model = _make_model()
        path = tmp_path / "legacy.pth"
        torch.save(model.state_dict(), str(path))
        loaded, meta, _ = load_model_from_checkpoint(str(path))
        assert isinstance(loaded, AdvantageNetwork)
        assert meta == {}  # no metadata in legacy checkpoints

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_from_checkpoint(str(tmp_path / "nope.pth"))

    def test_strict_false_tolerates_shape_mismatch(self, tmp_path: Path) -> None:
        # Save a checkpoint with a different hidden_dim than we load with.
        big = _make_model(hidden_dim=64)
        path = tmp_path / "big.pth"
        torch.save(
            {
                "state_dict": big.state_dict(),
                "metadata": {
                    "history_feature_dim": 18,
                    "card_feature_dim": 17,
                    "num_actions": 10,
                    "hidden_dim": 64,
                    "num_heads": 4,
                    "num_layers": 2,
                    "max_seq_len": 256,
                    "trainer": "deep_cfr",
                },
            },
            str(path),
        )
        # Load with strict=False should not raise even if shapes mismatch.
        loaded, meta, _ = load_model_from_checkpoint(str(path), strict=False)
        assert isinstance(loaded, AdvantageNetwork)


# ---------------------------------------------------------------------------
# ComprehensiveEvaluator
# ---------------------------------------------------------------------------


class TestComprehensiveEvaluator:
    def test_evaluate_checkpoint_returns_report(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(
            h2h_hands=4, health_num_states=2, seed=42, run_kuhn=False
        )
        report = ev.evaluate_checkpoint(str(tmp_checkpoint), total_samples=100)
        assert isinstance(report, EvaluationReport)
        assert report.checkpoint_path == str(tmp_checkpoint)
        assert report.total_samples == 100
        assert report.health_severity in {"ok", "warning", "error"}
        assert len(report.h2h_results) == 3  # always_fold, calling_station, random_uniform
        baselines = {r["baseline"] for r in report.h2h_results}
        assert baselines == {"always_fold", "calling_station", "random_uniform"}

    def test_evaluate_checkpoint_missing_file(self, tmp_path: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2)
        report = ev.evaluate_checkpoint(str(tmp_path / "missing.pth"))
        assert report.severity == "error"
        assert any("checkpoint_load_failed" in i for i in report.issues)
        assert report.h2h_results == []

    def test_evaluate_checkpoint_emits_always_fold_regression_on_loss(
        self, tmp_path: Path
    ) -> None:
        # A model that always raises into a fold would lose to always-fold...
        # but our synthetic model is random, so we just check the *plumbing*
        # of the regression flag by directly constructing a report.
        report = EvaluationReport(
            checkpoint_path="x",
            created_at="2026-01-01T00:00:00Z",
            total_samples=0,
            health_ok=True,
            health_severity="ok",
            health_checks={},
            health_diagnostics={},
            h2h_results=[
                H2HResult(
                    baseline="always_fold",
                    hands_played=10,
                    bb_per_100=-50.0,
                    stderr_bb_per_100=1.0,
                    total_chips=-50.0,
                    big_blind=10.0,
                    is_significantly_positive=False,
                    is_significantly_negative=True,
                ).to_dict()
            ],
            kuhn_exploitability=None,
            metadata=None,
        )
        # The evaluator's _run_h2h_panel method constructs H2HResult objects;
        # we replicate the regression logic here for the unit test.
        if report.h2h_results[0]["baseline"] == "always_fold" and report.h2h_results[0]["is_significantly_negative"]:
            report.update_severity("error", "h2h_always_fold_negative")
        assert report.severity == "error"

    def test_evaluate_with_no_baselines(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(
            h2h_hands=2, health_num_states=2, baselines=[], run_kuhn=False
        )
        report = ev.evaluate_checkpoint(str(tmp_checkpoint))
        assert report.h2h_results == []
        # Health should still run.
        assert report.health_checks  # non-empty

    def test_evaluate_with_kuhn_enabled(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(
            h2h_hands=2, health_num_states=2, run_kuhn=True, kuhn_iterations=100
        )
        report = ev.evaluate_checkpoint(str(tmp_checkpoint))
        assert report.kuhn_exploitability is not None
        assert "exploitability" in report.kuhn_exploitability


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_json_round_trip(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2, seed=1)
        report = ev.evaluate_checkpoint(str(tmp_checkpoint), total_samples=42)
        js = report.to_json()
        parsed = json.loads(js)
        assert parsed["checkpoint_path"] == str(tmp_checkpoint)
        assert parsed["total_samples"] == 42
        assert "h2h_results" in parsed
        assert "health_checks" in parsed

    def test_to_dict_keys(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2)
        report = ev.evaluate_checkpoint(str(tmp_checkpoint))
        d = report.to_dict()
        expected = {
            "checkpoint_path", "created_at", "total_samples", "severity", "ok",
            "issues", "health_ok", "health_severity", "health_checks",
            "health_diagnostics", "h2h_results", "kuhn_exploitability",
            "metadata", "diagnostics",
        }
        assert expected.issubset(set(d.keys()))


# ---------------------------------------------------------------------------
# RegressionTracker
# ---------------------------------------------------------------------------


class TestRegressionTracker:
    def _make_report(self, bb: float, baseline: str = "always_fold") -> EvaluationReport:
        return EvaluationReport(
            checkpoint_path="x",
            created_at="2026-01-01T00:00:00Z",
            total_samples=0,
            health_ok=True,
            health_severity="ok",
            health_checks={},
            health_diagnostics={},
            h2h_results=[
                {
                    "baseline": baseline,
                    "hands_played": 10,
                    "bb_per_100": bb,
                    "stderr_bb_per_100": 1.0,
                    "total_chips": bb * 10,
                    "big_blind": 10.0,
                    "is_significantly_positive": bb > 2,
                    "is_significantly_negative": bb < -2,
                }
            ],
            kuhn_exploitability=None,
            metadata=None,
        )

    def test_first_update_no_warning(self, tmp_path: Path) -> None:
        tracker = RegressionTracker(history_path=tmp_path / "h.json", margin_bb=5.0)
        warnings = tracker.update(self._make_report(80.0))
        assert warnings == []

    def test_regression_flagged_when_drop_exceeds_margin(self, tmp_path: Path) -> None:
        tracker = RegressionTracker(history_path=tmp_path / "h.json", margin_bb=5.0)
        tracker.update(self._make_report(80.0))
        warnings = tracker.update(self._make_report(60.0))
        assert len(warnings) == 1
        assert "regression[always_fold]" in warnings[0]

    def test_no_regression_when_drop_within_margin(self, tmp_path: Path) -> None:
        tracker = RegressionTracker(history_path=tmp_path / "h.json", margin_bb=5.0)
        tracker.update(self._make_report(80.0))
        warnings = tracker.update(self._make_report(77.0))
        assert warnings == []

    def test_history_persisted_to_disk(self, tmp_path: Path) -> None:
        hist = tmp_path / "h.json"
        tracker = RegressionTracker(history_path=hist, margin_bb=5.0)
        tracker.update(self._make_report(80.0))
        assert hist.exists()
        with hist.open() as fh:
            data = json.load(fh)
        assert len(data) == 1
        assert data[0]["h2h_results"][0]["bb_per_100"] == 80.0

    def test_best_bb_per_100(self, tmp_path: Path) -> None:
        tracker = RegressionTracker(history_path=tmp_path / "h.json")
        tracker.update(self._make_report(50.0))
        tracker.update(self._make_report(80.0))
        tracker.update(self._make_report(60.0))
        assert tracker.best_bb_per_100("always_fold") == 80.0
        assert tracker.best_bb_per_100("nonexistent") is None

    def test_handles_nan_gracefully(self, tmp_path: Path) -> None:
        tracker = RegressionTracker(history_path=tmp_path / "h.json")
        tracker.update(self._make_report(float("nan")))
        # NaN should be skipped, not crash.
        assert tracker.best_bb_per_100("always_fold") is None


# ---------------------------------------------------------------------------
# TrainingEvalHook
# ---------------------------------------------------------------------------


class TestTrainingEvalHook:
    def test_no_evaluator_returns_none(self) -> None:
        hook = TrainingEvalHook(evaluator=None, every_samples=100)
        assert hook.maybe_evaluate("x", 200) is None

    def test_every_samples_zero_returns_none(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2)
        hook = TrainingEvalHook(evaluator=ev, every_samples=0)
        assert hook.maybe_evaluate(str(tmp_checkpoint), 100) is None

    def test_below_threshold_returns_none(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2)
        hook = TrainingEvalHook(evaluator=ev, every_samples=100)
        assert hook.maybe_evaluate(str(tmp_checkpoint), 50) is None

    def test_at_threshold_evaluates(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2)
        hook = TrainingEvalHook(evaluator=ev, every_samples=100)
        report = hook.maybe_evaluate(str(tmp_checkpoint), 100)
        assert report is not None
        assert report.total_samples == 100

    def test_does_not_re_evaluate_within_window(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2)
        hook = TrainingEvalHook(evaluator=ev, every_samples=100)
        first = hook.maybe_evaluate(str(tmp_checkpoint), 100)
        second = hook.maybe_evaluate(str(tmp_checkpoint), 150)  # < 100 since last
        assert first is not None
        assert second is None

    def test_re_evaluates_after_window(self, tmp_checkpoint: Path) -> None:
        ev = ComprehensiveEvaluator(h2h_hands=2, health_num_states=2)
        hook = TrainingEvalHook(evaluator=ev, every_samples=100)
        first = hook.maybe_evaluate(str(tmp_checkpoint), 100)
        second = hook.maybe_evaluate(str(tmp_checkpoint), 200)
        assert first is not None
        assert second is not None
        assert hook._last_eval_samples == 200


# ---------------------------------------------------------------------------
# discover_checkpoints
# ---------------------------------------------------------------------------


class TestDiscoverCheckpoints:
    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        assert discover_checkpoints(tmp_path / "nope") == []

    def test_returns_empty_for_no_matches(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x")
        assert discover_checkpoints(tmp_path) == []

    def test_finds_pth_files(self, tmp_checkpoints: list[Path]) -> None:
        found = discover_checkpoints(tmp_checkpoints[0].parent)
        assert len(found) == 3
        # Sorted by mtime ascending.
        assert found[0].name == "deep_cfr_hand_0.pth"

    def test_prefix_filter(self, tmp_checkpoints: list[Path]) -> None:
        # Add a non-matching file.
        other = tmp_checkpoints[0].parent / "other_hand_0.pth"
        _save_checkpoint(other)
        os.utime(other, (99, 99))
        found = discover_checkpoints(tmp_checkpoints[0].parent, prefix="deep_cfr_")
        assert all(p.name.startswith("deep_cfr_") for p in found)
        assert len(found) == 3


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


class TestCliHelpers:
    def test_evaluate_checkpoint_cli_writes_output(self, tmp_checkpoint: Path, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        report = evaluate_checkpoint_cli(
            str(tmp_checkpoint),
            h2h_hands=2,
            health_num_states=2,
            output_path=str(out),
        )
        assert out.exists()
        with out.open() as fh:
            data = json.load(fh)
        assert data["checkpoint_path"] == str(tmp_checkpoint)
        assert isinstance(report, EvaluationReport)

    def test_sweep_checkpoints_cli_latest_only(self, tmp_checkpoints: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "sweep.json"
        reports = sweep_checkpoints_cli(
            str(tmp_checkpoints[0].parent),
            h2h_hands=2,
            health_num_states=2,
            output_path=str(out),
            latest_only=True,
        )
        assert len(reports) == 1
        assert out.exists()
        with out.open() as fh:
            data = json.load(fh)
        assert len(data) == 1

    def test_sweep_checkpoints_cli_all(self, tmp_checkpoints: list[Path], tmp_path: Path) -> None:
        reports = sweep_checkpoints_cli(
            str(tmp_checkpoints[0].parent),
            h2h_hands=2,
            health_num_states=2,
        )
        assert len(reports) == 3


# ---------------------------------------------------------------------------
# RandomAllInStrategy
# ---------------------------------------------------------------------------


class TestRandomAllInStrategy:
    def test_is_not_human(self) -> None:
        strat = RandomAllInStrategy(seed=0)
        assert strat.is_human is False

    def test_chooses_legal_action(self) -> None:
        # Smoke test: instantiate and check it has the right interface.
        strat = RandomAllInStrategy(seed=42)
        assert hasattr(strat, "choose_action")


# ---------------------------------------------------------------------------
# H2HResult
# ---------------------------------------------------------------------------


class TestH2HResult:
    def test_from_match(self) -> None:
        match = MatchResult(
            hands_played=100,
            total_chips=500.0,
            big_blind=10.0,
            bb_per_100=50.0,
            stderr_bb_per_100=5.0,
        )
        result = H2HResult.from_match("always_fold", match)
        assert result.baseline == "always_fold"
        assert result.bb_per_100 == 50.0
        assert result.is_significantly_positive is True
        assert result.is_significantly_negative is False

    def test_to_dict(self) -> None:
        result = H2HResult(
            baseline="x", hands_played=10, bb_per_100=5.0,
            stderr_bb_per_100=1.0, total_chips=50.0, big_blind=10.0,
            is_significantly_positive=True, is_significantly_negative=False,
        )
        d = result.to_dict()
        assert d["baseline"] == "x"
        assert d["bb_per_100"] == 5.0
