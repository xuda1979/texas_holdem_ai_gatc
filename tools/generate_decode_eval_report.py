#!/usr/bin/env python3
"""Generate a consolidated 解码与评估 (decode & evaluation) report.

This helper script executes the test targets that exercise the decoding
pipelines (state representation helpers) and the evaluation utilities.  The
results are written to a Markdown report so contributors can quickly inspect
the outcome after modifying the related code paths.

Usage
-----

.. code-block:: bash

   python tools/generate_decode_eval_report.py

An optional ``--output`` flag customises the destination Markdown file.  The
report includes a compact table summarising pass/fail information for each
target alongside collapsible sections containing the raw logs.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


@dataclasses.dataclass(slots=True)
class CommandTarget:
    """Definition of a command that should be executed."""

    label: str
    command: Sequence[str]


@dataclasses.dataclass(slots=True)
class TargetGroup:
    """Grouping of related :class:`CommandTarget` instances."""

    heading: str
    targets: tuple[CommandTarget, ...]


@dataclasses.dataclass(slots=True)
class CommandResult:
    """Captured outcome of executing a :class:`CommandTarget`."""

    target: CommandTarget
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def _run_target(target: CommandTarget) -> CommandResult:
    """Execute ``target`` and capture its output."""

    completed = subprocess.run(  # noqa: PLW1510 - return code inspected manually
        list(target.command),
        check=False,
        text=True,
        capture_output=True,
    )
    return CommandResult(
        target=target,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _write_report(
    groups: Iterable[TargetGroup],
    results: dict[str, CommandResult],
    output_path: Path,
) -> None:
    lines: list[str] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines.append("# 解码与评估结果报告")
    lines.append("")
    lines.append(f"生成时间：{timestamp}")
    lines.append("")

    for group in groups:
        lines.append(f"## {group.heading}")
        lines.append("")
        lines.append("| 测试项 | 状态 | 命令 |")
        lines.append("| ------ | ---- | ---- |")
        for target in group.targets:
            result = results[target.label]
            status = "✅ 通过" if result.succeeded else "❌ 失败"
            lines.append(
                f"| {target.label} | {status} | `{_format_command(target.command)}` |"
            )
        lines.append("")

        for target in group.targets:
            result = results[target.label]
            lines.append(f"### {group.heading} · {target.label}")
            lines.append("")
            lines.append("```text")
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if stdout:
                lines.append(stdout)
            if stderr:
                if stdout:
                    lines.append("")
                lines.append("[stderr]")
                lines.append(stderr)
            if not stdout and not stderr:
                lines.append("(no output)")
            lines.append("```")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_targets() -> tuple[TargetGroup, ...]:
    python = sys.executable
    decode_targets = TargetGroup(
        heading="解码 (State Decoding) 检查",
        targets=(
            CommandTarget(
                label="状态表示单元测试",
                command=(python, "-m", "pytest", "tests/test_state_representation.py", "-q"),
            ),
            CommandTarget(
                label="游戏状态一致性测试",
                command=(python, "-m", "pytest", "tests/test_state.py", "-q"),
            ),
        ),
    )

    evaluation_targets = TargetGroup(
        heading="评估 (Evaluation) 检查",
        targets=(
            CommandTarget(
                label="AI GTO 分析模块",
                command=(python, "-m", "pytest", "tests/evaluation", "-q"),
            ),
            CommandTarget(
                label="评估基准套件",
                command=(python, "-m", "pytest", "tests/eval", "-q"),
            ),
        ),
    )

    return (decode_targets, evaluation_targets)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/decode_eval_report.md"),
        help="Destination path for the generated Markdown report.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    groups = _build_targets()

    results: dict[str, CommandResult] = {}
    all_passed = True
    for group in groups:
        for target in group.targets:
            result = _run_target(target)
            results[target.label] = result
            all_passed &= result.succeeded

    _write_report(groups, results, args.output)
    print(f"Report written to {args.output}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
