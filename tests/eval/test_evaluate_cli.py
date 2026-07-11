"""Tests for the ``poker_ai.cli.evaluate`` command-line interface.

These tests invoke the CLI's ``main`` function directly with synthetic
arguments so they don't depend on a trained model being present.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("POKER_AI_ALLOW_LOCAL_MODEL_WRITES", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.cli.evaluate import main


def _save_checkpoint(path: Path) -> Path:
    model = AdvantageNetwork(
        history_feature_dim=18, card_feature_dim=17, hidden_dim=32,
        num_heads=4, num_layers=2, num_actions=10,
    )
    payload = {
        "state_dict": model.state_dict(),
        "policy_net_state_dict": model.state_dict(),
        "metadata": {
            "history_feature_dim": 18, "card_feature_dim": 17, "num_actions": 10,
            "hidden_dim": 32, "num_heads": 4, "num_layers": 2,
            "max_seq_len": 256, "trainer": "deep_cfr",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return path


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    return _save_checkpoint(tmp_path / "model.pth")


class TestEvaluateCLI:
    def test_health_subcommand(self, ckpt: Path, capsys) -> None:
        rc = main(["health", "--path", str(ckpt), "--health-states", "2"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "HealthReport" in captured.out
        assert "severity=ok" in captured.out

    def test_health_writes_json_output(self, ckpt: Path, tmp_path: Path, capsys) -> None:
        out = tmp_path / "health.json"
        rc = main(["health", "--path", str(ckpt), "--health-states", "2", "--output", str(out)])
        assert rc == 0
        assert out.exists()
        with out.open() as fh:
            data = json.load(fh)
        assert "checks" in data
        assert data["checkpoint_path"] == str(ckpt)

    def test_checkpoint_subcommand(self, ckpt: Path, tmp_path: Path, capsys) -> None:
        out = tmp_path / "report.json"
        rc = main([
            "checkpoint", "--path", str(ckpt),
            "--h2h-hands", "2", "--health-states", "2",
            "--output", str(out),
        ])
        assert rc == 0
        assert out.exists()
        with out.open() as fh:
            data = json.load(fh)
        assert data["checkpoint_path"] == str(ckpt)
        assert "h2h_results" in data
        assert len(data["h2h_results"]) == 3

    def test_sweep_subcommand_latest_only(self, ckpt: Path, tmp_path: Path) -> None:
        # Make a second checkpoint so sweep has multiple to choose from.
        second = _save_checkpoint(ckpt.parent / "model2.pth")
        os.utime(second, (999, 999))  # make it newest
        out = tmp_path / "sweep.json"
        rc = main([
            "sweep", "--dir", str(ckpt.parent),
            "--h2h-hands", "2", "--health-states", "2",
            "--latest-only", "--output", str(out),
        ])
        assert rc == 0
        assert out.exists()
        with out.open() as fh:
            data = json.load(fh)
        assert len(data) == 1

    def test_sweep_no_checkpoints_returns_2(self, tmp_path: Path, capsys) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = main(["sweep", "--dir", str(empty)])
        assert rc == 2

    def test_baselines_subcommand(self, ckpt: Path, capsys) -> None:
        rc = main([
            "baselines", "--path", str(ckpt),
            "--hands", "2", "--no-health",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "always_fold" in captured.out
        assert "calling_station" in captured.out

    def test_h2h_subcommand(self, ckpt: Path, tmp_path: Path, capsys) -> None:
        # Use the same checkpoint for both sides.
        rc = main([
            "h2h", "--model-a", str(ckpt), "--model-b", str(ckpt),
            "--hands", "2",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Match:" in captured.out

    def test_h2h_missing_model_returns_1(self, ckpt: Path, tmp_path: Path, capsys) -> None:
        rc = main([
            "h2h", "--model-a", str(ckpt), "--model-b", str(tmp_path / "nope.pth"),
            "--hands", "2",
        ])
        assert rc == 1

    def test_health_missing_file_returns_1(self, tmp_path: Path, capsys) -> None:
        rc = main(["health", "--path", str(tmp_path / "nope.pth")])
        assert rc == 1
