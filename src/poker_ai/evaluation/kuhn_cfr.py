"""Kuhn-poker CFR sanity harness with exact exploitability evaluation.

The :class:`~poker_ai.evaluation.kuhn_poker.KuhnCFR` trainer is the legacy
chance-sampling implementation used by the comprehensive evaluation harness.
Its information sets use letter cards (``J``/``Q``/``K``) and dict-form
strategies.  :mod:`poker_ai.games.kuhn` provides an *exact* best-response
exploitability evaluator for int-card (``1``/``2``/``3``) list-form
strategies.  This module bridges the two so the harness reports the real
Nash-convergence metric instead of a synthetic value.
"""

from __future__ import annotations

from poker_ai.games.kuhn import exploitability as _exact_exploitability

from .kuhn_poker import CARDS, KuhnCFR

# Letter card -> int card used by poker_ai.games.kuhn (J=1, Q=2, K=3).
_CARD_TO_INT = {card: str(i + 1) for i, card in enumerate(CARDS)}


def kuhn_strategy_table(trainer: KuhnCFR) -> dict[str, list[float]]:
    """Convert a trained :class:`KuhnCFR` into a ``games.kuhn`` strategy table.

    The exact best-response evaluator in :mod:`poker_ai.games.kuhn` expects
    ``Dict[str, List[float]]`` keyed by ``f"{card}{history}"`` where ``card``
    is an int in ``{1, 2, 3}`` and the list is ``[prob_pass, prob_bet]``.
    """
    table: dict[str, list[float]] = {}
    for info_key, node in trainer.nodes.items():
        card_letter = info_key[0]
        history = info_key[1:]
        avg = node.average_strategy()
        table[f"{_CARD_TO_INT.get(card_letter, card_letter)}{history}"] = [
            float(avg["p"]),
            float(avg["b"]),
        ]
    return table


def kuhn_exploitability(trainer: KuhnCFR) -> float:
    """Exact exploitability (in chips) of a trained Kuhn strategy profile.

    Uses the exact best-response evaluator from :mod:`poker_ai.games.kuhn`;
    ``0.0`` exactly at a Nash equilibrium.
    """
    return float(_exact_exploitability(kuhn_strategy_table(trainer)))


def train_kuhn_cfr(iterations: int = 20_000, seed: int = 7) -> KuhnCFR:
    """
    Trains CFR on Kuhn Poker and returns the trainer.
    Useful as a tiny convergence sanity test harness.
    """
    k = KuhnCFR()
    k.train(iterations=iterations, seed=seed)
    return k


__all__ = ["KuhnCFR", "train_kuhn_cfr", "kuhn_strategy_table", "kuhn_exploitability"]
