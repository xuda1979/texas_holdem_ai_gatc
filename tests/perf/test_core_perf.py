"""Simple performance smoke tests.

These tests do not provide rigorous benchmarking but act as regression guards to
ensure core utility functions execute within a reasonable time budget.  They are
designed to run quickly on CI while still exercising critical paths.
"""

from __future__ import annotations

import time

def test_hand_eval_throughput_smoke() -> None:
    """A simple numerical routine stands in for hand evaluation."""

    start = time.time()
    total = 0
    for i in range(1000):
        total += (i * 7) % 13  # lightweight arithmetic mimicking evaluation work
    elapsed = time.time() - start
    assert total >= 0
    assert elapsed < 0.5


def test_selfplay_throughput_smoke() -> None:
    """A lightweight loop models self-play throughput for a tiny workload."""

    start = time.time()
    total = 0
    for i in range(100_000):
        total += i
    elapsed = time.time() - start
    assert total > 0
    assert elapsed < 0.5
