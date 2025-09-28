#!/usr/bin/env python3
"""Texas Hold'em AI test orchestrator.

This helper wraps ``pytest`` so contributors can run the complete automated
suite with a single command.  By default the entire ``tests/`` tree (including
unit, integration, CLI, and service tests) is executed.  Optional flags enable
additional slow checks such as the GUI smoke tests, end-to-end scenarios, and
performance harnesses.

Examples
--------
Run the default fast suite::

    ./run_all_tests.py

Include the slower end-to-end and GUI checks::

    ./run_all_tests.py --include-e2e --include-gui

Run every target the script knows about::

    ./run_all_tests.py --include-all
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence

@dataclass
class TestTarget:
    """Definition of an individual test command."""

    name: str
    command: Sequence[str]


@dataclass
class TestResult:
    """Outcome of executing a :class:`TestTarget`."""

    target: TestTarget
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def _run_target(target: TestTarget) -> TestResult:
    """Execute ``target`` and capture its output."""

    print(f"\n=== Running {target.name} ===")
    print("$", " ".join(shlex.quote(part) for part in target.command))
    try:
        completed = subprocess.run(
            list(target.command),
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        message = f"Failed to launch {target.name}: {exc}"
        print(message)
        return TestResult(target=target, returncode=127, stdout="", stderr=message)

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())

    if completed.returncode == 0:
        print(f"✔ {target.name} completed successfully.")
    else:
        print(f"✖ {target.name} failed with exit code {completed.returncode}.")

    return TestResult(
        target=target,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-e2e",
        action="store_true",
        help="Also run end-to-end tests located under tests/e2e.",
    )
    parser.add_argument(
        "--include-gui",
        action="store_true",
        help="Run GUI smoke tests that rely on headless Tkinter rendering.",
    )
    parser.add_argument(
        "--include-perf",
        action="store_true",
        help="Run performance regression smoke tests under tests/perf.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Shortcut to enable all optional suites (e2e, gui, perf).",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Additional arguments appended to each pytest invocation.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_matrix(args: argparse.Namespace) -> List[TestTarget]:
    pytest_args = shlex.split(args.pytest_args)
    matrix: List[TestTarget] = [
        TestTarget(
            name="pytest (core suite)",
            command=[sys.executable, "-m", "pytest", "-q", *pytest_args],
        )
    ]

    include_e2e = args.include_all or args.include_e2e
    include_gui = args.include_all or args.include_gui
    include_perf = args.include_all or args.include_perf

    if include_e2e:
        matrix.append(
            TestTarget(
                name="pytest (end-to-end)",
                command=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/e2e",
                    "-q",
                    *pytest_args,
                ],
            )
        )

    if include_perf:
        matrix.append(
            TestTarget(
                name="pytest (performance)",
                command=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/perf",
                    "-q",
                    *pytest_args,
                ],
            )
        )

    if include_gui:
        gui_targets = [
            "test_complete_gui.py",
            "test_comprehensive_gui.py",
            "test_gui_core.py",
            "test_gui_faithfulness.py",
            "test_gui_human_vs_ai.py",
            "test_gui_integration.py",
            "test_gui.py",
            "test_card_images_gui.py",
        ]
        matrix.append(
            TestTarget(
                name="pytest (gui smoke)",
                command=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *gui_targets,
                    *pytest_args,
                ],
            )
        )

    return matrix


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    matrix = _build_matrix(args)

    results = [_run_target(target) for target in matrix]

    print("\n=== Summary ===")
    passed = 0
    for result in results:
        status = "PASS" if result.succeeded else "FAIL"
        print(f"{status:>4}  {result.target.name}")
        if result.succeeded:
            passed += 1

    total = len(results)
    print(f"\nCompleted {total} command(s); {passed} succeeded, {total - passed} failed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
