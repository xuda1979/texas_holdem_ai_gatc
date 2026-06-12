"""Rigorous Nash-equilibrium convergence guarantees for the CFR implementations.

These tests verify the theoretical contract of CFR (Zinkevich et al., 2007):
the *average* strategy profile converges to a Nash equilibrium, i.e.

  1. exploitability eps_T = (br0 + br1) / 2 decays toward 0,
  2. the average game value converges to the known Kuhn value -1/18,
  3. the learned strategy lies in the known one-parameter (alpha) family of
     Kuhn Nash equilibria,
  4. a correct best response can never lose against any fixed profile
     (br_i >= v_i at every checkpoint).

They intentionally use the exact infoset-level best response, which is the
only sound convergence measure: a per-deal ("clairvoyant") best response does
not vanish at equilibrium.
"""

from __future__ import annotations

import itertools

import pytest

from poker_ai.games import KuhnTrainer, best_response_value, exploitability
from poker_ai.games.kuhn import ACTIONS

GAME_VALUE_P1 = -1.0 / 18.0


def _average_game_value_p1(strategy: dict[str, list[float]]) -> float:
    """Expected value for player 0 when both players follow ``strategy``."""

    def rec(cards: list[int], history: str) -> float:
        payoff = KuhnTrainer._terminal_utility(cards, history, 0)
        if payoff is not None:
            return payoff
        current = len(history) % 2
        probs = strategy.get(f"{cards[current]}{history}", [0.5, 0.5])
        return sum(p * rec(cards, history + a) for p, a in zip(probs, ACTIONS))

    total = sum(rec(list(cards), "") for cards in itertools.permutations([1, 2, 3], 2))
    return total / 6.0


@pytest.fixture(scope="module")
def trained() -> tuple[KuhnTrainer, list[tuple[int, float]]]:
    """Train tabular CFR, recording exploitability at decade checkpoints."""
    trainer = KuhnTrainer()
    history: list[tuple[int, float]] = []
    done = 0
    for checkpoint in (100, 1000, 10000):
        trainer.train(checkpoint - done)
        done = checkpoint
        eps = exploitability(trainer.get_average_strategy())
        history.append((checkpoint, eps))
    return trainer, history


def test_exploitability_decays_to_zero(trained) -> None:
    _, history = trained
    epsilons = [eps for _, eps in history]
    # Exploitability is non-negative for any profile and must decay.
    assert all(eps >= 0.0 for eps in epsilons)
    assert epsilons[1] < epsilons[0]
    assert epsilons[2] < epsilons[1]
    # CFR guarantees eps = O(1/sqrt(T)); on Kuhn 10k full-tree iterations
    # reach well below 0.005 chips.
    assert epsilons[-1] < 0.005


def test_best_response_never_loses(trained) -> None:
    trainer, _ = trained
    strategy = trainer.get_average_strategy()
    value_p1 = _average_game_value_p1(strategy)
    # A correct best response is at least as good as following the profile.
    assert best_response_value(strategy, 0) >= value_p1 - 1e-12
    assert best_response_value(strategy, 1) >= -value_p1 - 1e-12


def test_game_value_converges_to_minus_one_eighteenth(trained) -> None:
    trainer, _ = trained
    value = _average_game_value_p1(trainer.get_average_strategy())
    assert value == pytest.approx(GAME_VALUE_P1, abs=0.005)


def test_average_strategy_in_kuhn_equilibrium_family(trained) -> None:
    """Kuhn equilibria form a one-parameter family (alpha in [0, 1/3]):

    P1: bet J with alpha, bet Q never, bet K with 3*alpha,
        after check-bet: fold J, call Q with alpha + 1/3, call K always.
    P2 (facing a check): bet J with 1/3, check Q, bet K always.
    P2 (facing a bet):   fold J, call Q with 1/3, call K always.
    Strategy entries are [P(pass), P(bet/call)].
    """
    trainer, _ = trained
    s = trainer.get_average_strategy()
    alpha = s["1"][1]
    tol = 0.05

    assert -1e-9 <= alpha <= 1.0 / 3.0 + tol
    assert s["2"][1] == pytest.approx(0.0, abs=tol)
    assert s["3"][1] == pytest.approx(3.0 * alpha, abs=tol)
    assert s["1pb"][1] == pytest.approx(0.0, abs=tol)
    assert s["2pb"][1] == pytest.approx(alpha + 1.0 / 3.0, abs=tol)
    assert s["3pb"][1] == pytest.approx(1.0, abs=tol)

    assert s["1p"][1] == pytest.approx(1.0 / 3.0, abs=tol)
    assert s["2p"][1] == pytest.approx(0.0, abs=tol)
    assert s["3p"][1] == pytest.approx(1.0, abs=tol)
    assert s["1b"][1] == pytest.approx(0.0, abs=tol)
    assert s["2b"][1] == pytest.approx(1.0 / 3.0, abs=tol)
    assert s["3b"][1] == pytest.approx(1.0, abs=tol)


def test_clairvoyant_strategies_remain_exploitable() -> None:
    """Sanity check of the metric itself: a pure always-bet profile is
    exploitable, and the uniform profile is exploitable, so the measure must
    be strictly positive away from equilibrium."""
    trainer = KuhnTrainer()
    trainer.train(1)
    uniform = {k: [0.5, 0.5] for k in trainer.get_average_strategy()}
    assert exploitability(uniform) > 0.05
    always_bet = {k: [0.0, 1.0] for k in uniform}
    assert exploitability(always_bet) > 0.05
