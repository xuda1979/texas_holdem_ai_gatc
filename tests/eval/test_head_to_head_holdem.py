"""Tests for the full-Hold'em head-to-head evaluation harness."""

from __future__ import annotations

import random

import pytest

from poker_ai.evaluation.head_to_head import (
    AlwaysFoldStrategy,
    CallingStationStrategy,
    MatchResult,
    play_hand,
    play_match,
)
from poker_ai.gui.playStrategy import RandomAIStrategy


def test_play_hand_is_zero_sum_and_deterministic_per_seed() -> None:
    random.seed(0)
    payoffs_1 = play_hand([CallingStationStrategy(), CallingStationStrategy()], deck_seed=42)
    payoffs_2 = play_hand([CallingStationStrategy(), CallingStationStrategy()], deck_seed=42)
    assert payoffs_1 == payoffs_2
    assert sum(payoffs_1) == pytest.approx(0.0)


def test_duplicate_self_match_scores_zero() -> None:
    """With duplicate dealing, a deterministic strategy vs itself nets 0."""
    result = play_match(
        CallingStationStrategy(), CallingStationStrategy(), num_hands=40, seed=7
    )
    assert result.bb_per_100 == pytest.approx(0.0, abs=1e-9)


def test_calling_station_beats_always_fold() -> None:
    """A caller collects the folder's blinds; sign and significance must hold."""
    result = play_match(CallingStationStrategy(), AlwaysFoldStrategy(), num_hands=60, seed=1)
    assert result.bb_per_100 > 10.0
    # Mirror match: same strategies with roles swapped must negate the score.
    mirrored = play_match(AlwaysFoldStrategy(), CallingStationStrategy(), num_hands=60, seed=1)
    assert mirrored.bb_per_100 == pytest.approx(-result.bb_per_100, abs=1e-9)


def test_random_strategy_beats_always_fold_significantly() -> None:
    random.seed(123)
    result = play_match(RandomAIStrategy(), AlwaysFoldStrategy(), num_hands=100, seed=5)
    assert result.is_significantly_positive()


def test_match_result_significance_helpers() -> None:
    strong = MatchResult(
        hands_played=100, total_chips=500.0, big_blind=10.0, bb_per_100=50.0,
        stderr_bb_per_100=5.0,
    )
    noisy = MatchResult(
        hands_played=100, total_chips=50.0, big_blind=10.0, bb_per_100=5.0,
        stderr_bb_per_100=20.0,
    )
    assert strong.is_significantly_positive()
    assert not strong.is_significantly_negative()
    assert not noisy.is_significantly_positive()
    assert not noisy.is_significantly_negative()
