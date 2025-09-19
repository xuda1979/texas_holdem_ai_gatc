"""Utilities for managing Google Cloud training jobs.

This module exposes a small command line interface that helps launch and
control GPU or TPU training infrastructure for the poker agent.  The commands
simply wrap the `gcloud` CLI to keep the workflow self contained inside the
repository so developers do not need to memorise the individual `gcloud`
invocations.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def ensure_gcloud_available() -> None:
    """Verify that the `gcloud` CLI is present on the system.

    The helper raises ``SystemExit`` with a friendly message if the
    requirement is not satisfied.  This makes the command feel similar to the
    behaviour of ``argparse`` when required arguments are missing.
    """

    if shutil.which("gcloud") is None:
        message = (
            "The gcloud CLI could not be found. Please install the Google Cloud "
            "SDK and ensure the `gcloud` executable is available on your PATH."
        )
        raise SystemExit(message)


def _run_command(command: Iterable[str]) -> None:
    """Run ``command`` with ``subprocess.run`` and surface errors gracefully."""

    command_list: List[str] = list(command)
    try:
        subprocess.run(command_list, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
        joined = " ".join(shlex.quote(part) for part in command_list)
        raise SystemExit(
            f"Command failed with exit code {exc.returncode}: {joined}"
        ) from exc


def _repo_root() -> Path:
    """Return the root of the repository."""

    return Path(__file__).resolve().parents[3]


def create_instance(args: argparse.Namespace) -> None:
    """Create a GPU or TPU instance using ``gcloud``."""

    ensure_gcloud_available()

    if args.accelerator == "gpu":
        accelerator_type = args.gpu_type or "nvidia-tesla-t4"
        accelerator_flag = f"type={accelerator_type},count={args.gpu_count}"
        command = [
            "gcloud",
            "compute",
            "instances",
            "create",
            args.name,
            "--project",
            args.project,
            "--zone",
            args.zone,
            "--machine-type",
            args.machine_type,
            "--accelerator",
            accelerator_flag,
            "--maintenance-policy",
            "TERMINATE",
            "--restart-on-failure",
        ]

        if args.image:
            command.extend(["--image", args.image])
        elif args.image_family:
            command.extend(["--image-family", args.image_family])
            if args.image_project:
                command.extend(["--image-project", args.image_project])

        _run_command(command)
    else:  # TPU
        tpu_type = args.tpu_type or "v4-8"
        command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "create",
            args.name,
            "--project",
            args.project,
            "--zone",
            args.zone,
            "--accelerator-type",
            tpu_type,
            "--version",
            args.tpu_version,
        ]
        _run_command(command)


def run_training(args: argparse.Namespace) -> None:
    """Synchronise the repository and execute the training command remotely."""

    ensure_gcloud_available()

    repo_root = _repo_root()
    source_path = os.path.join(str(repo_root), ".")
    remote_path = f"{args.name}:~/poker-ai"

    if args.accelerator == "gpu":
        sync_command = [
            "gcloud",
            "compute",
            "scp",
            "--recurse",
            source_path,
            remote_path,
            "--project",
            args.project,
            "--zone",
            args.zone,
        ]
        ssh_command = [
            "gcloud",
            "compute",
            "ssh",
            args.name,
            "--project",
            args.project,
            "--zone",
            args.zone,
            "--command",
            _remote_training_command(args),
        ]
    else:
        sync_command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            "--recurse",
            source_path,
            remote_path,
            "--project",
            args.project,
            "--zone",
            args.zone,
        ]
        ssh_command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            args.name,
            "--project",
            args.project,
            "--zone",
            args.zone,
            "--command",
            _remote_training_command(args),
        ]

    _run_command(sync_command)
    _run_command(ssh_command)


def _remote_training_command(args: argparse.Namespace) -> str:
    """Return the remote command string used by ``run_training``."""

    command_prefix = "cd ~/poker-ai && "

    if args.bucket:
        bucket_export = shlex.quote(args.bucket)
        command_prefix = f"export CHECKPOINT_BUCKET={bucket_export} && " + command_prefix

    return command_prefix + args.command


def delete_instance(args: argparse.Namespace) -> None:
    """Delete the GPU instance or TPU VM."""

    ensure_gcloud_available()

    if args.accelerator == "gpu":
        command = [
            "gcloud",
            "compute",
            "instances",
            "delete",
            args.name,
            "--quiet",
            "--project",
            args.project,
            "--zone",
            args.zone,
        ]
    else:
        command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            args.name,
            "--quiet",
            "--project",
            args.project,
            "--zone",
            args.zone,
        ]

    _run_command(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for launching Google Cloud training jobs."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud project that hosts the compute resources.",
    )
    parser.add_argument(
        "--zone",
        required=True,
        help="Compute zone to deploy to (for example us-central1-a).",
    )
    parser.add_argument(
        "--name",
        default="poker-ai-trainer",
        help="Name of the VM or TPU to manage.",
    )
    parser.add_argument(
        "--machine-type",
        default="n1-standard-8",
        help="Machine type to request for GPU based training.",
    )
    parser.add_argument(
        "--accelerator",
        choices=("gpu", "tpu"),
        default="gpu",
        help="Type of accelerator to provision (gpu or tpu).",
    )
    parser.add_argument(
        "--bucket",
        help="Optional Cloud Storage bucket used to persist checkpoints.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create the instance")
    create_parser.add_argument(
        "--gpu-type",
        help="Fully qualified GPU type when using accelerator=gpu.",
    )
    create_parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs to attach to the VM.",
    )
    create_parser.add_argument(
        "--image",
        help="Optional custom image for the VM.",
    )
    create_parser.add_argument(
        "--image-family",
        default="pytorch-latest-gpu",
        help="Image family for deep learning VM images.",
    )
    create_parser.add_argument(
        "--image-project",
        default="deeplearning-platform-release",
        help="Project hosting the image family.",
    )
    create_parser.add_argument(
        "--tpu-type",
        help="TPU type identifier when using accelerator=tpu.",
    )
    create_parser.add_argument(
        "--tpu-version",
        default="tpu-vm-base",
        help="TPU runtime version.",
    )
    create_parser.set_defaults(func=create_instance)

    run_parser = subparsers.add_parser(
        "run", help="Synchronise the repo and run a training command"
    )
    run_parser.add_argument(
        "command",
        help="Training command to execute remotely, e.g. 'python -m poker_ai.cli.train'.",
    )
    run_parser.set_defaults(func=run_training)

    delete_parser = subparsers.add_parser("delete", help="Delete the instance")
    delete_parser.set_defaults(func=delete_instance)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ``run`` is implemented as ``set_defaults(func=...)``.
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(2)

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
