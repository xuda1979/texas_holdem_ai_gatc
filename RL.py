"""Convenience launcher for the Poker AI training CLI.

This wrapper simplifies running :mod:`poker_ai.cli.train` on cloud
instances where the repository has been cloned directly (for example a
Google Compute Engine VM).  It mirrors :mod:`run_training.py` but adds
quality-of-life features such as GPU selection and pass-through CLI
options so it can be used from orchestration scripts.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _build_train_command(args: argparse.Namespace) -> list[str]:
    command: list[str] = [sys.executable, "-m", "poker_ai.cli.train"]
    if args.algorithm:
        command.extend(["--algorithm", args.algorithm])
    if args.num_hands is not None:
        command.extend(["--num-hands", str(args.num_hands)])
    if args.save_model_every is not None:
        command.extend(["--save-model-every", str(args.save_model_every)])
    if args.config:
        command.extend(["--config", args.config])
    if args.extra_train_args:
        command.extend(args.extra_train_args)
    if args.enable_gpus:
        command.append("--gpus")
    if args.enable_npus:
        command.append("--npus")
    if args.enable_tpu:
        command.append("--tpu")
    return command


def _prepare_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.world_size:
        env["WORLD_SIZE"] = str(args.world_size)
    return env


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch poker_ai.cli.train with optional accelerator helpers.",
    )
    parser.add_argument("--algorithm", default="deep_cfr", help="Training algorithm to run.")
    parser.add_argument("--num-hands", type=int, help="Number of hands to simulate.")
    parser.add_argument(
        "--save-model-every",
        type=int,
        help="Save the model every N hands (overrides configuration).",
    )
    parser.add_argument("--config", help="Optional configuration YAML file.")
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to poker_ai.cli.train.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help="Explicit CUDA_VISIBLE_DEVICES value (e.g. '0,1,2,3').",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="Optional WORLD_SIZE environment override for torch distributed setups.",
    )
    accelerators = parser.add_argument_group("accelerator toggles")
    accelerators.add_argument(
        "--enable-gpus",
        action="store_true",
        help="Request GPU training via the train CLI --gpus flag.",
    )
    accelerators.add_argument(
        "--enable-npus",
        action="store_true",
        help="Request NPU training via the train CLI --npus flag.",
    )
    accelerators.add_argument(
        "--enable-tpu",
        action="store_true",
        help="Request TPU training via the train CLI --tpu flag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    command = _build_train_command(args)
    env = _prepare_env(args)
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[RL] $ {printable}")
    if args.dry_run:
        return
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])
