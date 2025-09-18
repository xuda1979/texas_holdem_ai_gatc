"""GATC Poker helpers: rules-safe side pots and hand evaluation wrappers."""

from __future__ import annotations

from typing import Iterable

from .pots import SidePot, compute_side_pots

__all__ = ["best_of", "evaluate_hand", "compute_side_pots", "SidePot"]


def evaluate_hand(hole: Iterable[str], board: Iterable[str]) -> int:
    """Proxy for :func:`gatc_poker.handeval.evaluate_hand` with lazy import."""

    from .handeval import evaluate_hand as _evaluate_hand

    return _evaluate_hand(list(hole), list(board))


def best_of(players_hole: dict[int, list[str]], board: Iterable[str]) -> list[int]:
    """Proxy for :func:`gatc_poker.handeval.best_of` with lazy import."""

    from .handeval import best_of as _best_of

    return _best_of(players_hole, list(board))
