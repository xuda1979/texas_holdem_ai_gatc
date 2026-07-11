"""Tests for the in-training evaluation hook integration in poker_ai.cli.train.

These tests verify that ``_build_comprehensive_eval_hook`` correctly reads
the ``evaluation`` section of the config and constructs a working
``TrainingEvalHook`` (or ``None`` when evaluation is disabled).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POKER_AI_ALLOW_LOCAL_MODEL_WRITES", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from poker_ai.cli.train import _build_comprehensive_eval_hook
from poker_ai.evaluation.comprehensive import TrainingEvalHook


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_eval_hook")


class TestBuildComprehensiveEvalHook:
    def test_no_evaluation_section_returns_none(self, logger: logging.Logger) -> None:
        assert _build_comprehensive_eval_hook(config={}, device="cpu", logger=logger) is None

    def test_empty_evaluation_section_returns_none(self, logger: logging.Logger) -> None:
        cfg = {"evaluation": {}}
        assert _build_comprehensive_eval_hook(config=cfg, device="cpu", logger=logger) is None

    def test_every_samples_zero_returns_none(self, logger: logging.Logger) -> None:
        cfg = {"evaluation": {"every_samples": 0}}
        assert _build_comprehensive_eval_hook(config=cfg, device="cpu", logger=logger) is None

    def test_negative_every_samples_returns_none(self, logger: logging.Logger) -> None:
        cfg = {"evaluation": {"every_samples": -5}}
        assert _build_comprehensive_eval_hook(config=cfg, device="cpu", logger=logger) is None

    def test_valid_config_returns_hook(self, logger: logging.Logger) -> None:
        cfg = {
            "evaluation": {
                "every_samples": 100,
                "h2h_hands": 50,
                "health_num_states": 8,
                "seed": 42,
                "emit_events": True,
                "stop_on_error": True,
            }
        }
        hook = _build_comprehensive_eval_hook(config=cfg, device="cpu", logger=logger)
        assert isinstance(hook, TrainingEvalHook)
        assert hook.every_samples == 100
        assert hook.stop_on_error is True
        assert hook.emit_events is True
        assert hook.evaluator is not None
        assert hook.evaluator.h2h_hands == 50
        assert hook.evaluator.health_num_states == 8
        assert hook.evaluator.seed == 42
        assert hook.tracker is not None

    def test_non_dict_evaluation_section_returns_none(self, logger: logging.Logger) -> None:
        cfg = {"evaluation": "not a dict"}
        assert _build_comprehensive_eval_hook(config=cfg, device="cpu", logger=logger) is None

    def test_partial_config_uses_defaults(self, logger: logging.Logger) -> None:
        cfg = {"evaluation": {"every_samples": 50}}
        hook = _build_comprehensive_eval_hook(config=cfg, device="cpu", logger=logger)
        assert isinstance(hook, TrainingEvalHook)
        # Defaults should be applied.
        assert hook.evaluator.h2h_hands == 200
        assert hook.evaluator.health_num_states == 32
