"""Helper script for training on Google Cloud GPU or TPU accelerators.

This entry point is intended to run *inside* a Google Cloud VM or TPU VM
instance after the repository has been synchronised.  It can optionally install
PyTorch/torch-xla wheels that match the accelerator and then invoke the
standard :mod:`poker_ai.cli.train` command with the correct device flags.

Example usage for a GPU VM::

    python tools/google_accelerator_train.py \
        --accelerator gpu \
        --install-deps \
        --num-hands 2000 \
        --algorithm deep_cfr

Example usage for a TPU VM::

    python tools/google_accelerator_train.py \
        --accelerator tpu \
        --install-deps \
        --num-hands 2000 \
        --algorithm deep_cfr

The script prints every command before executing it so that it is easy to audit
in production environments.  Pass ``--dry-run`` to only display the commands
without running them.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

# Ensure the project source tree is available when the project has not been
# installed as a Python package.  This mirrors ``run_training.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _run_command(command: Iterable[str], *, env: Mapping[str, str] | None = None, dry_run: bool = False) -> None:
    """Execute ``command`` with ``subprocess.run`` while echoing the command line.

    Parameters
    ----------
    command:
        Sequence of command line arguments to execute.
    env:
        Optional environment variables to pass to ``subprocess.run``.
    dry_run:
        If ``True`` the command is only printed and not executed.  This is
        useful for validating the actions before running them on a newly
        provisioned VM.
    """

    command_list = list(command)
    printable = " ".join(shlex.quote(part) for part in command_list)
    print(f"[google-accelerator] $ {printable}")
    if dry_run:
        return
    subprocess.run(command_list, check=True, env=dict(env) if env is not None else None)


def _install_base_requirements(*, dry_run: bool) -> None:
    """Install the repository requirements using the active Python interpreter."""

    requirements_path = REPO_ROOT / "requirements.txt"
    if not requirements_path.exists():
        raise SystemExit(f"Could not find requirements.txt at {requirements_path}.")

    # Upgrade pip to ensure wheel resolution works for accelerator specific
    # packages (for example CUDA wheels).
    _run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], dry_run=dry_run)
    _run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], dry_run=dry_run)


def _install_gpu_packages(args: argparse.Namespace, *, dry_run: bool) -> None:
    """Install PyTorch GPU wheels that match the configured CUDA runtime."""

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch=={args.torch_version}",
        f"torchvision=={args.torchvision_version}",
        f"torchaudio=={args.torchaudio_version}",
    ]
    if args.torch_index_url:
        command.extend(["--index-url", args.torch_index_url])
    if args.extra_pip_args:
        command.extend(args.extra_pip_args)
    _run_command(command, dry_run=dry_run)


def _install_tpu_packages(args: argparse.Namespace, *, dry_run: bool) -> None:
    """Install PyTorch + torch_xla wheels for TPU training."""

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch=={args.torch_version}",
        f"torchvision=={args.torchvision_version}",
        f"torch-xla=={args.torch_xla_version}",
    ]
    if args.tpu_wheel_url:
        command.extend(["-f", args.tpu_wheel_url])
    if args.extra_pip_args:
        command.extend(args.extra_pip_args)
    _run_command(command, dry_run=dry_run)


def _build_training_command(args: argparse.Namespace) -> list[str]:
    """Construct the ``poker_ai.cli.train`` invocation for the selected accelerator."""

    command = [sys.executable, "-m", "poker_ai.cli.train"]
    if args.algorithm:
        command.extend(["--algorithm", args.algorithm])
    if args.num_hands is not None:
        command.extend(["--num-hands", str(args.num_hands)])
    if args.config:
        command.extend(["--config", args.config])
    if args.save_model_every is not None:
        command.extend(["--save-model-every", str(args.save_model_every)])
    if args.accelerator == "gpu":
        command.append("--gpus")
    else:
        command.append("--tpu")
    if args.train_args:
        command.extend(shlex.split(args.train_args))
    return command


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the helper script."""

    parser = argparse.ArgumentParser(
        description="Bootstrap training on Google Cloud GPU or TPU accelerators.",
    )
    parser.add_argument(
        "--accelerator",
        choices=("gpu", "tpu"),
        required=True,
        help="Type of accelerator available on the VM.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install requirements and accelerator specific PyTorch wheels before training.",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        help="Number of hands to simulate during training (overrides config).",
    )
    parser.add_argument(
        "--algorithm",
        default="deep_cfr",
        help="Training algorithm to use (default: deep_cfr).",
    )
    parser.add_argument(
        "--config",
        help="Optional path to a configuration YAML file on the remote VM.",
    )
    parser.add_argument(
        "--save-model-every",
        type=int,
        help="Save the model every N hands (overrides config).",
    )
    parser.add_argument(
        "--train-args",
        help="Additional arguments to append to the training command (quoted string).",
    )
    parser.add_argument(
        "--bucket",
        help="Cloud Storage bucket used for checkpoint exports (sets CHECKPOINT_BUCKET).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--extra-pip-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the accelerator wheel pip install command.",
    )

    gpu_group = parser.add_argument_group("GPU options")
    gpu_group.add_argument(
        "--torch-version",
        default="2.2.1",
        help="PyTorch version to install for accelerators (default: 2.2.1).",
    )
    gpu_group.add_argument(
        "--torchvision-version",
        default="0.17.1",
        help="torchvision version (default: 0.17.1).",
    )
    gpu_group.add_argument(
        "--torchaudio-version",
        default="2.2.1",
        help="torchaudio version (default: 2.2.1).",
    )
    gpu_group.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cu118",
        help="Custom index URL used to download CUDA enabled wheels.",
    )

    tpu_group = parser.add_argument_group("TPU options")
    tpu_group.add_argument(
        "--torch-xla-version",
        default="2.2.1",
        help="torch_xla version to install when ``--accelerator tpu`` is selected.",
    )
    tpu_group.add_argument(
        "--tpu-wheel-url",
        default="https://storage.googleapis.com/tpu-pytorch/wheels/colab.html",
        help="Wheel index used for TPU compatible packages.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.install_deps:
        _install_base_requirements(dry_run=args.dry_run)
        if args.accelerator == "gpu":
            _install_gpu_packages(args, dry_run=args.dry_run)
        else:
            _install_tpu_packages(args, dry_run=args.dry_run)

    command = _build_training_command(args)
    env = os.environ.copy()
    if args.bucket:
        env["CHECKPOINT_BUCKET"] = args.bucket
    _run_command(command, env=env, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])
