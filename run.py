"""Helper script to launch self-play or evaluation loops.

The repository contains a rich command-line interface under
``poker_ai.cli``.  This thin wrapper mirrors the convenience of
``run_training.py`` and :mod:`RL.py` by delegating to
:mod:`poker_ai.cli.self_play`.  It keeps the repository importable when
running directly from a Google Compute Engine VM without installing the
package system wide.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch poker_ai.cli.self_play with pass-through arguments.",
    )
    parser.add_argument(
        "--config",
        help="Optional configuration YAML file to load before starting self-play.",
    )
    parser.add_argument(
        "--extra-self-play-args",
        nargs=argparse.REMAINDER,
        help="Additional CLI arguments forwarded to poker_ai.cli.self_play.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "poker_ai.cli.self_play"]
    if args.config:
        command.extend(["--config", args.config])
    if args.extra_self_play_args:
        command.extend(args.extra_self_play_args)
    return command


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    command = build_command(args)
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"[run] $ {printable}")
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(sys.argv[1:])
