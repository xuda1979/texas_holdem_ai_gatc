"""Compatibility entrypoint so `pytest tests/test_gui_human_vs_ai.py` just works."""
from __future__ import annotations

from tests.gui.test_gui_human_vs_ai import (
    test_human_vs_ai_setup as _test_human_vs_ai_setup,
    test_invalid_setups as _test_invalid_setups,
    test_turn_management as _test_turn_management,
)


def test_human_vs_ai_setup() -> None:
    _test_human_vs_ai_setup()


def test_invalid_setups() -> None:
    _test_invalid_setups()


def test_turn_management() -> None:
    _test_turn_management()
