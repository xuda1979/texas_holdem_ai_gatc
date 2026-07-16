#!/usr/bin/env python3
"""Analyze advantage_net vs policy_net weight norms across Deep CFR checkpoints.

This tool was added to diagnose the calling_station regression pattern where
the model peaks around sample 1000-1100 and then degrades (see
``docs/SESSION_20260713_TRAJECTORY.md``). By comparing the L2 norms and layer-wise
statistics of the advantage and policy networks across the training trajectory,
we can distinguish between several hypotheses:

* **Advantage-net explosion**: the advantage net's weights grow unboundedly
  while the policy net stays stable -> regret targets overshoot.
* **Policy-net drift**: the policy net drifts away from the exploitation
  strategy the advantage net discovered.
* **Divergence**: both nets' norms grow together, indicating general instability.

Usage (on ai3, where checkpoints live)::

    python3 tools/analyze_checkpoint_norms.py \\
        --checkpoint-dir models \\
        --pattern 'deep_cfr_time_*.pth' \\
        --output models/checkpoint_norms.json

Then fetch the JSON locally and plot, or read the summary table printed to
stdout. The tool is pure-CPU and only reads checkpoints (it never trains), so
it is safe to run on a live training box.

The output JSON is a list of records, one per checkpoint, sorted by sample
number::

    [
      {
        "samples": 512,
        "path": "models/deep_cfr_time_512.pth",
        "advantage_norm": 12.34,
        "policy_norm": 9.87,
        "advantage_per_layer": {"...": ...},
        "policy_per_layer": {"...": ...},
        "advantage_max_abs": 1.23,
        "policy_max_abs": 0.98
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch  # noqa: E402


SAMPLE_RE = re.compile(r"(\d+)\.pth$")


def _extract_samples(path: str) -> int | None:
    """Extract the sample count from a checkpoint filename."""
    m = SAMPLE_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def _state_dict_norm(state_dict: dict[str, torch.Tensor]) -> tuple[float, dict[str, float], float]:
    """Return (global L2 norm, per-layer L2 norms, max absolute weight)."""
    total_sq = 0.0
    max_abs = 0.0
    per_layer: dict[str, float] = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            continue
        flat = tensor.detach().float().flatten()
        layer_norm = float(torch.linalg.norm(flat).item())
        per_layer[name] = round(layer_norm, 6)
        total_sq += layer_norm * layer_norm
        layer_max = float(flat.abs().max().item())
        if layer_max > max_abs:
            max_abs = layer_max
    return round(total_sq**0.5, 6), per_layer, round(max_abs, 6)


def analyze_checkpoint(path: str) -> dict[str, Any] | None:
    """Load a checkpoint and compute norm statistics for both networks."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARN: failed to load {path}: {exc}", file=sys.stderr)
        return None

    adv_key = "state_dict" if "state_dict" in state else None
    pol_key = "policy_net_state_dict" if "policy_net_state_dict" in state else None
    if adv_key is None:
        # Legacy checkpoints may be a bare state_dict.
        if all(isinstance(v, torch.Tensor) for v in state.values()):
            adv_sd = state
            pol_sd = {}
        else:
            print(f"  WARN: no state_dict in {path}", file=sys.stderr)
            return None
    else:
        adv_sd = state[adv_key]
        pol_sd = state.get(pol_key, {}) if pol_key else {}

    adv_norm, adv_per_layer, adv_max = _state_dict_norm(adv_sd)
    if pol_sd:
        pol_norm, pol_per_layer, pol_max = _state_dict_norm(pol_sd)
    else:
        pol_norm, pol_per_layer, pol_max = 0.0, {}, 0.0

    samples = _extract_samples(path)
    return {
        "samples": samples,
        "path": path,
        "advantage_norm": adv_norm,
        "policy_norm": pol_norm,
        "advantage_per_layer": adv_per_layer,
        "policy_per_layer": pol_per_layer,
        "advantage_max_abs": adv_max,
        "policy_max_abs": pol_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint-dir",
        default="models",
        help="Directory containing checkpoints (default: models)",
    )
    parser.add_argument(
        "--pattern",
        default="deep_cfr_time_*.pth",
        help="Glob pattern for checkpoints (default: deep_cfr_time_*.pth)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON results to this path (default: stdout summary only)",
    )
    parser.add_argument(
        "--include-best",
        action="store_true",
        help="Also include best_calling_station_*.pth checkpoints",
    )
    args = parser.parse_args()

    patterns = [args.pattern]
    if args.include_best:
        patterns.append("best/best_calling_station_*.pth")

    paths: list[str] = []
    for pat in patterns:
        full = os.path.join(args.checkpoint_dir, pat)
        paths.extend(sorted(glob.glob(full)))

    if not paths:
        print(f"No checkpoints matching {patterns} in {args.checkpoint_dir}", file=sys.stderr)
        return 1

    print(f"Analyzing {len(paths)} checkpoints in {args.checkpoint_dir}...", file=sys.stderr)
    records: list[dict[str, Any]] = []
    for p in paths:
        rec = analyze_checkpoint(p)
        if rec is not None:
            records.append(rec)

    # Sort by sample count (None sorts last).
    records.sort(key=lambda r: (r["samples"] is None, r["samples"] or 0))

    # Print a compact summary table.
    print()
    print(f"{'samples':>8}  {'adv_norm':>10}  {'pol_norm':>10}  {'ratio a/p':>10}  {'adv_max':>9}  {'pol_max':>9}  path")
    print("-" * 90)
    for r in records:
        s = r["samples"] if r["samples"] is not None else "?"
        ratio = (r["advantage_norm"] / r["policy_norm"]) if r["policy_norm"] > 0 else float("inf")
        print(
            f"{str(s):>8}  {r['advantage_norm']:>10.4f}  {r['policy_norm']:>10.4f}  "
            f"{ratio:>10.4f}  {r['advantage_max_abs']:>9.4f}  {r['policy_max_abs']:>9.4f}  {os.path.basename(r['path'])}"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nWrote {len(records)} records to {args.output}", file=sys.stderr)

    # Detect runaway growth.
    if len(records) >= 2 and records[0]["samples"] is not None and records[-1]["samples"] is not None:
        first, last = records[0], records[-1]
        if first["advantage_norm"] > 0:
            growth = last["advantage_norm"] / first["advantage_norm"]
            print(f"\nAdvantage-net norm growth: {first['advantage_norm']:.4f} -> {last['advantage_norm']:.4f} ({growth:.2f}x)")
        if first["policy_norm"] > 0:
            growth = last["policy_norm"] / first["policy_norm"]
            print(f"Policy-net norm growth:    {first['policy_norm']:.4f} -> {last['policy_norm']:.4f} ({growth:.2f}x)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
