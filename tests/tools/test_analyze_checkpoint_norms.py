"""Tests for the checkpoint norm analysis tool.

Verifies that ``tools.analyze_checkpoint_norms`` correctly loads Deep CFR
checkpoints, computes L2 norms for the advantage and policy networks, and
produces a sorted summary. Uses synthetic checkpoints so the test is fast and
hermetic.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from tools import analyze_checkpoint_norms as acn


def _make_checkpoint(path: str, adv_scale: float = 1.0, pol_scale: float = 1.0) -> None:
    """Write a synthetic Deep CFR checkpoint to ``path``."""
    payload = {
        "state_dict": {
            "layer.weight": torch.eye(4) * adv_scale,
            "layer.bias": torch.zeros(4),
        },
        "policy_net_state_dict": {
            "layer.weight": torch.eye(4) * pol_scale,
            "layer.bias": torch.zeros(4),
        },
        "model_config": {"trainer": "deep_cfr"},
    }
    torch.save(payload, path)


def test_extract_samples_from_filename() -> None:
    assert acn._extract_samples("models/deep_cfr_time_1024.pth") == 1024
    assert acn._extract_samples("models/best/best_calling_station_512.pth") == 512
    assert acn._extract_samples("models/random_name.pth") is None


def test_state_dict_norm_identity_matrix() -> None:
    # A 4x4 identity has L2 norm sqrt(4) = 2.0
    sd = {"weight": torch.eye(4)}
    norm, per_layer, max_abs = acn._state_dict_norm(sd)
    assert norm == pytest.approx(2.0, rel=1e-5)
    assert per_layer["weight"] == pytest.approx(2.0, rel=1e-5)
    assert max_abs == pytest.approx(1.0, rel=1e-5)


def test_state_dict_norm_ignores_non_floating_point() -> None:
    sd = {
        "weight": torch.eye(4),
        "num_batches_tracked": torch.tensor(5, dtype=torch.long),
    }
    norm, per_layer, max_abs = acn._state_dict_norm(sd)
    assert norm == pytest.approx(2.0, rel=1e-5)
    assert "num_batches_tracked" not in per_layer


def test_analyze_checkpoint_full_payload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "deep_cfr_time_100.pth")
        _make_checkpoint(path, adv_scale=2.0, pol_scale=1.0)
        rec = acn.analyze_checkpoint(path)
    assert rec is not None
    assert rec["samples"] == 100
    # 4x4 identity * 2.0 has norm 4.0
    assert rec["advantage_norm"] == pytest.approx(4.0, rel=1e-5)
    assert rec["policy_norm"] == pytest.approx(2.0, rel=1e-5)
    assert rec["advantage_max_abs"] == pytest.approx(2.0, rel=1e-5)
    assert "layer.weight" in rec["advantage_per_layer"]


def test_analyze_checkpoint_bare_state_dict() -> None:
    """Legacy checkpoints that are a bare state_dict (no wrapper) still work."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "deep_cfr_time_50.pth")
        torch.save({"weight": torch.eye(4)}, path)
        rec = acn.analyze_checkpoint(path)
    assert rec is not None
    assert rec["samples"] == 50
    assert rec["advantage_norm"] == pytest.approx(2.0, rel=1e-5)
    assert rec["policy_norm"] == 0.0


def test_main_produces_sorted_summary(tmp_path: object) -> None:
    tmp = str(tmp_path)
    for samples, adv, pol in [(300, 1.0, 1.0), (100, 3.0, 2.0), (200, 2.0, 1.5)]:
        _make_checkpoint(
            os.path.join(tmp, f"deep_cfr_time_{samples}.pth"),
            adv_scale=adv,
            pol_scale=pol,
        )
    out = os.path.join(tmp, "out.json")
    rc = acn.main.__wrapped__ if hasattr(acn.main, "__wrapped__") else None
    # Invoke main via the module's argparse by setting sys.argv.
    import sys

    orig_argv = sys.argv
    sys.argv = ["analyze_checkpoint_norms", "--checkpoint-dir", tmp, "--output", out]
    try:
        code = acn.main()
    finally:
        sys.argv = orig_argv
    assert code == 0
    with open(out) as f:
        records = json.load(f)
    assert [r["samples"] for r in records] == [100, 200, 300]
    assert records[0]["advantage_norm"] > records[1]["advantage_norm"] > records[2]["advantage_norm"]
