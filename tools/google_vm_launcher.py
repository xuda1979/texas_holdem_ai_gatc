"""Launch RL and evaluation scripts on Google Cloud GPU VMs.

This helper focuses on A100-equipped Google Compute Engine instances
but works for any VM that exposes multiple CUDA devices.  It can install
project dependencies, expose a configurable subset of GPUs via the
``CUDA_VISIBLE_DEVICES`` environment variable, and then run ``RL.py``
(training) alongside ``run.py`` (self-play monitoring).

Example::

    python tools/google_vm_launcher.py \
        --num-gpus 4 \
        --rl-extra "--num-hands 20000 --save-model-every 500" \
        --run-extra "--extra-self-play-args --episodes 5" \
        --log-dir /var/log/poker-ai
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DEFAULT_RL = str(REPO_ROOT / "RL.py")
DEFAULT_RUN = str(REPO_ROOT / "run.py")


class ProcessHandle:
    def __init__(self, name: str, process: subprocess.Popen[str], thread: threading.Thread):
        self.name = name
        self.process = process
        self.thread = thread

    def terminate(self, timeout: float = 10.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.thread.join(timeout=timeout)


def _ensure_log_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _install_requirements(dry_run: bool) -> None:
    requirements = REPO_ROOT / "requirements.txt"
    if not requirements.exists():
        raise SystemExit("requirements.txt not found; run from the repository root.")
    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
    ]
    for command in commands:
        printable = " ".join(shlex.quote(part) for part in command)
        print(f"[setup] $ {printable}")
        if dry_run:
            continue
        subprocess.run(command, check=True)


def _format_env(env: Mapping[str, str]) -> str:
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))


def _start_process(
    name: str,
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    log_dir: Path | None,
    dry_run: bool,
) -> ProcessHandle | None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[{name}] $ {printable}")
    if env:
        print(f"[{name}] env { _format_env(env) }")
    if dry_run:
        return None
    log_file: Path | None = None
    if log_dir is not None:
        log_file = log_dir / f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env) if env is not None else None,
    )

    def _pump_output() -> None:
        try:
            if process.stdout is None:
                return
            with (log_file.open("a", encoding="utf-8") if log_file is not None else None) as log:
                for line in process.stdout:
                    decorated = f"[{name}] {line.rstrip()}"
                    print(decorated)
                    if log is not None:
                        log.write(decorated + "\n")
        finally:
            if process.stdout is not None:
                process.stdout.close()

    thread = threading.Thread(target=_pump_output, name=f"{name}-logger", daemon=True)
    thread.start()
    return ProcessHandle(name, process, thread)


def _build_cuda_visible_devices(num_gpus: int | None, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if num_gpus is None:
        return None
    if num_gpus <= 0:
        return None
    return ",".join(str(idx) for idx in range(num_gpus))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate RL.py and run.py on a multi-GPU Google VM.",
    )
    parser.add_argument("--rl-script", default=DEFAULT_RL, help="Path to the RL training wrapper.")
    parser.add_argument("--run-script", default=DEFAULT_RUN, help="Path to the self-play wrapper.")
    parser.add_argument(
        "--rl-extra",
        default="",
        help="Additional CLI options appended to the RL script (quoted string).",
    )
    parser.add_argument(
        "--run-extra",
        default="",
        help="Additional CLI options appended to run.py (quoted string).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Expose this many GPUs via CUDA_VISIBLE_DEVICES (default: all visible).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help="Explicit CUDA_VISIBLE_DEVICES value overriding --num-gpus.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="WORLD_SIZE environment value forwarded to RL.py for torch distributed setups.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Optional directory where stdout/stderr logs are tee'd.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install requirements.txt before launching any process.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only launch RL.py and skip run.py entirely.",
    )
    parser.add_argument(
        "--run-delay",
        type=float,
        default=15.0,
        help="Seconds to wait after starting training before launching run.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _split_extra(extra: str) -> list[str]:
    if not extra:
        return []
    return shlex.split(extra)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    log_dir = _ensure_log_dir(args.log_dir)
    if args.install_deps:
        _install_requirements(dry_run=args.dry_run)

    cuda_visible = _build_cuda_visible_devices(args.num_gpus, args.cuda_visible_devices)
    shared_env: dict[str, str] = {}
    if cuda_visible:
        shared_env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    if args.world_size:
        shared_env["WORLD_SIZE"] = str(args.world_size)

    rl_command = [sys.executable, args.rl_script]
    rl_command.extend(_split_extra(args.rl_extra))
    run_command = [sys.executable, args.run_script]
    run_command.extend(_split_extra(args.run_extra))

    handles: list[ProcessHandle] = []
    try:
        rl_handle = _start_process(
            "RL",
            rl_command,
            env=shared_env,
            log_dir=log_dir,
            dry_run=args.dry_run,
        )
        if rl_handle is None:
            return
        handles.append(rl_handle)

        if not args.skip_run:
            if not args.dry_run and args.run_delay > 0:
                time.sleep(args.run_delay)
            run_handle = _start_process(
                "run",
                run_command,
                env=shared_env,
                log_dir=log_dir,
                dry_run=args.dry_run,
            )
            if run_handle is not None:
                handles.append(run_handle)

        # Wait for the RL process; terminate run.py if it is still running.
        exit_code = rl_handle.process.wait()
        print(f"[RL] exited with status {exit_code}")
    except KeyboardInterrupt:
        print("[launcher] Caught Ctrl+C, terminating processes...")
    finally:
        for handle in handles:
            handle.terminate()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])
