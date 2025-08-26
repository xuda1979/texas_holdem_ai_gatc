from __future__ import annotations
from typing import List, Tuple, Any

"""
Evaluator adapter.

Prefers the fast 'eval7' library if available. Falls back to a *very* naive
tie-breaker on high-card patterns so unit tests can run without native deps.
Replace this with Treys or your preferred C/C++-backed evaluator in prod.
"""

try:
    import eval7
except Exception:  # pragma: no cover - fallback path
    eval7 = None


def _cards_to_eval7(cards: List[str]) -> Any:
    assert eval7 is not None
    return [eval7.Card(c) for c in cards]


def best5_rank_key(hand: List[str], board: List[str]) -> Tuple[int, int]:
    """Return a comparable rank key; higher is better.

    If eval7 is present, we use its 7-card ranking. Otherwise, a naive
    surrogate that sorts by (distinct ranks, highest rank, length).
    The fallback is NOT poker-correct; it's only to keep examples runnable.
    """
    if eval7:
        cards = _cards_to_eval7(hand + board)
        score = eval7.evaluate(cards)  # higher is better in eval7
        return (score, len(set(hand + board)))
    # Fallback: not correct, but deterministic
    ranks = "23456789TJQKA"
    order = {r: i for i, r in enumerate(ranks)}
    only_ranks = [c[0] for c in hand + board]
    highest = max(order[r] for r in only_ranks)
    distinct = len(set(only_ranks))
    return (distinct * 100 + highest, len(only_ranks))
