"""Colab automation for training the Texas Hold'em AI with Drive checkpoints.

This script mirrors the one-off Colab notebook snippet that prepares the
environment, keeps checkpoints in Google Drive, and optionally mirrors them to
Google Cloud Storage.

Usage (in Google Colab):
    %run tools/colab_pro_plus_trainer.py

The script assumes it is executed inside a Colab runtime. It mounts Google
Drive, clones or updates the repository, installs dependencies, restores the
latest checkpoint if present, then launches the trainer while teeing the output
into a timestamped log file stored in Drive.
"""

import glob
import os
import pathlib
import shlex
import subprocess
import sys
import time

from google.colab import drive

REPO_URL = "https://github.com/xuda1979/texas_holdem_ai_gatc.git"
BRANCH = "main"
ALGORITHM = "deep_cfr"
NUM_HANDS = 5000
SAVE_EVERY = 100
EXTRA_TRAIN_ARGS = ""

USE_GCS_MIRROR = False
GCP_PROJECT = "YOUR_GCP_PROJECT_ID"
GCS_BUCKET = "gs://YOUR_BUCKET_NAME/holdem"

# Packages that are nice-to-have but frequently unavailable in the Colab
# environment.  `torch-xla` is only required when training on TPUs, so we do not
# want installation failures for that wheel to stop the rest of the setup.
OPTIONAL_REQUIREMENTS = {"torch-xla"}

CONTENT_ROOT = "/content"
REPO_DIR = os.path.join(CONTENT_ROOT, "texas_holdem_ai_gatc")
DRIVE_ROOT = "/content/drive/MyDrive/texas_holdem_ai_gatc"
SYMLINK_DIRS = ("trained_models", "logs", "reports")


def run_command(command, *, cwd=None, check=True):
    """Execute *command* and stream the output in real time."""
    printable = command if isinstance(command, str) else " ".join(command)
    print(f"\n[RUN] {printable}\n")
    process = subprocess.Popen(
        command if isinstance(command, list) else shlex.split(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        stdout_lines.append(line)
    process.wait()
    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    return process.returncode, "".join(stdout_lines)


def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: pathlib.Path) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            for child in path.iterdir():
                if child.is_dir() and not child.is_symlink():
                    remove_path(child)
                else:
                    child.unlink()
            path.rmdir()
        else:
            path.unlink()


def _normalise_requirement(req: str) -> str:
    """Return the package name portion of a requirement specifier."""

    req = req.split(";", 1)[0]
    req = req.split("[", 1)[0]
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if separator in req:
            return req.split(separator, 1)[0].strip().lower()
    return req.strip().lower()


def install_requirements() -> None:
    requirements_path = pathlib.Path(REPO_DIR) / "requirements.txt"

    def load_requirements() -> list[str]:
        requirements: list[str] = []
        with requirements_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                requirements.append(stripped)
        return requirements

    code, output = run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=False,
    )
    if code == 0:
        return

    print("Initial dependency install failed; attempting a fallback install.")
    requirements = load_requirements()

    filtered_requirements: list[str] = []
    skipped: list[str] = []
    for requirement in requirements:
        name = _normalise_requirement(requirement)
        if name in OPTIONAL_REQUIREMENTS:
            skipped.append(requirement)
        else:
            filtered_requirements.append(requirement)

    if not skipped:
        raise subprocess.CalledProcessError(
            code,
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            output,
        )

    print("Skipping optional packages that failed to install:")
    for item in skipped:
        print("  -", item)

    run_command([sys.executable, "-m", "pip", "install", *filtered_requirements])


def prepare_environment() -> None:
    drive.mount("/content/drive", force_remount=True)

    for directory in SYMLINK_DIRS:
        ensure_dir(pathlib.Path(DRIVE_ROOT) / directory)

    os.chdir(CONTENT_ROOT)
    if not os.path.exists(REPO_DIR):
        run_command(["git", "clone", "-b", BRANCH, REPO_URL])

    os.chdir(REPO_DIR)
    run_command(["git", "pull"], check=False)
    run_command(["git", "submodule", "update", "--init", "--recursive"], check=False)

    run_command([sys.executable, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    install_requirements()

    import torch

    print("Torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())

    for directory in SYMLINK_DIRS:
        target = pathlib.Path(DRIVE_ROOT) / directory
        link = pathlib.Path(REPO_DIR) / directory
        if link.exists() or link.is_symlink():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                remove_path(link)
        link.symlink_to(target)


def resolve_latest_checkpoint() -> str:
    ckpts = sorted(
        glob.glob(f"{DRIVE_ROOT}/trained_models/*.pth"),
        key=os.path.getmtime,
    )
    if ckpts:
        latest = ckpts[-1]
        print("Resuming from:", latest)
        return latest
    print("No checkpoint found; starting fresh.")
    return ""


def build_training_command(load_model_path: str) -> tuple[list[str], str]:
    import torch

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"{DRIVE_ROOT}/logs/train_{timestamp}.log"

    cmd = [
        sys.executable,
        "-m",
        "poker_ai.cli.train",
        "--algorithm",
        ALGORITHM,
        "--num-hands",
        str(NUM_HANDS),
        "--save-model-every",
        str(SAVE_EVERY),
    ]

    if load_model_path:
        cmd.extend(["--load-model-path", load_model_path])

    if torch.cuda.is_available():
        cmd.append("--gpus")

    if EXTRA_TRAIN_ARGS:
        cmd.extend(shlex.split(EXTRA_TRAIN_ARGS))

    print("Running:\n", " ".join(cmd))
    print("Logging to:", log_path)
    return cmd, log_path


def stream_training(cmd: list[str], log_path: str) -> int:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
    process.wait()
    print("\nReturn code:", process.returncode)
    run_command(["ls", "-lh", "trained_models"], check=False)
    return process.returncode


def mirror_to_gcs() -> None:
    if not USE_GCS_MIRROR:
        return

    from google.colab import auth

    auth.authenticate_user()
    run_command(["gcloud", "config", "set", "project", GCP_PROJECT])
    run_command(["gsutil", "-m", "rsync", "-r", "trained_models", f"{GCS_BUCKET}/trained_models"])
    run_command(["gsutil", "-m", "rsync", "-r", "logs", f"{GCS_BUCKET}/logs"])
    print("Mirrored to:", GCS_BUCKET)


def main() -> None:
    prepare_environment()
    load_path = resolve_latest_checkpoint()
    cmd, log_path = build_training_command(load_path)
    exit_code = stream_training(cmd, log_path)
    mirror_to_gcs()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
