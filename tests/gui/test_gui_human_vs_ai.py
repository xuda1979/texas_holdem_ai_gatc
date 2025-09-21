"""Pytest-compatible wrapper around the legacy GUI human-vs-AI smoke tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

LEGACY_TEST_PATH = Path(__file__).resolve().parents[2] / "test_gui_human_vs_ai.py"


def _load_legacy_module() -> ModuleType:
    """Load the legacy stand-alone test script as a module."""
    spec = importlib.util.spec_from_file_location(
        "legacy_gui_human_vs_ai", LEGACY_TEST_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy test module at {LEGACY_TEST_PATH!s}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY_MODULE = _load_legacy_module()


def test_human_vs_ai_setup() -> None:
    LEGACY_MODULE.test_human_vs_ai_setup()


def test_invalid_setups() -> None:
    LEGACY_MODULE.test_invalid_setups()


def test_turn_management() -> None:
    LEGACY_MODULE.test_turn_management()
