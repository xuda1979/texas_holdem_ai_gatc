"""Tests for the Leduc poker game and CFR/exploitability solver."""

from __future__ import annotations

import itertools

import pytest

from poker_ai.games.leduc import (
    ACTIONS,
    DECK,
    LeducGame,
    LeducState,
    best_response_value,
    exploitability,
    leduc_cfr,
)


# ---------------------------------------------------------------------------
# Game-tree correctness
# ---------------------------------------------------------------------------


def test_deck_has_six_cards():
    assert len(DECK) == 6
    ranks = {c[0] for c in DECK}
    assert ranks == {"J", "Q", "K"}


def test_deal_initial_state():
    game = LeducGame()
    state = game.deal(hole_cards=("Js", "Qh"), public_card="Ks")
    assert state.hole == ("Js", "Qh")
    assert state.dealt_public == "Ks"
    assert state.public is None  # not revealed preflop
    assert state.round == 0
    assert state.contributions == (1, 1)
    assert state.pot == 2
    assert state.current_player == 0
    assert not state.terminal


def test_legal_actions_preflop():
    game = LeducGame()
    state = game.deal(hole_cards=("Js", "Qh"), public_card="Ks")
    assert game.legal_actions(state) == ["f", "c", "r"]


def test_raise_cap_removes_raise_option():
    game = LeducGame()
    state = game.deal(hole_cards=("Js", "Qh"), public_card="Ks")
    # Player 0 raises.
    s1 = game.apply_action(state, "r")
    assert s1.raises_this_round == 1
    assert game.legal_actions(s1) == ["f", "c", "r"]
    # Player 1 raises (cap reached).
    s2 = game.apply_action(s1, "r")
    assert s2.raises_this_round == 2
    # Now only fold/call remain.
    assert game.legal_actions(s2) == ["f", "c"]


def test_fold_is_terminal_and_winner_is_other():
    game = LeducGame()
    state = game.deal(hole_cards=("Js", "Qh"), public_card="Ks")
    s1 = game.apply_action(state, "f")
    assert game.is_terminal(s1)
    # Player 0 folded; player 1 wins the pot (2 antes).
    assert game.terminal_utility(s1, 0) == -1.0  # lost the ante
    assert game.terminal_utility(s1, 1) == 1.0


def test_call_preflop_reveals_public_and_starts_postflop():
    game = LeducGame()
    state = game.deal(hole_cards=("Js", "Qh"), public_card="Ks")
    # Player 0 calls, player 1 calls -> preflop closes.
    s1 = game.apply_action(state, "c")
    s2 = game.apply_action(s1, "c")
    assert s2.round == 1
    assert s2.public == " Ks".strip()
    assert not game.is_terminal(s2)
    # Standard Leduc keeps a fixed first actor (player 0) both rounds.
    assert s2.current_player == 0


def test_showdown_pair_beats_higher_rank():
    game = LeducGame()
    # Player 0 has Js, public is Js -> pair.  Player 1 has Ks (higher).
    state = game.deal(hole_cards=("Js", "Ks"), public_card="Jh")
    # Play to showdown: call-call preflop, call-call postflop.
    s = game.apply_action(state, "c")
    s = game.apply_action(s, "c")
    assert s.round == 1
    s = game.apply_action(s, "c")
    s = game.apply_action(s, "c")
    assert game.is_terminal(s)
    # Pair of Jacks beats a lone King.
    assert game.terminal_utility(s, 0) > 0
    assert game.terminal_utility(s, 1) < 0


def test_showdown_higher_rank_wins_no_pair():
    game = LeducGame()
    state = game.deal(hole_cards=("Ks", "Js"), public_card="Qh")
    s = game.apply_action(state, "c")
    s = game.apply_action(s, "c")
    s = game.apply_action(s, "c")
    s = game.apply_action(s, "c")
    assert game.is_terminal(s)
    # King high beats Jack high (no pairs).
    assert game.terminal_utility(s, 0) > 0
    assert game.terminal_utility(s, 1) < 0


def test_illegal_action_raises():
    game = LeducGame()
    state = game.deal(hole_cards=("Js", "Qh"), public_card="Ks")
    with pytest.raises(ValueError):
        game.apply_action(state, "x")


# ---------------------------------------------------------------------------
# CFR convergence (the key property: exploitability -> 0)
# ---------------------------------------------------------------------------


def test_cfr_uniform_strategy_has_positive_exploitability():
    """A uniform-random strategy should be exploitable."""
    # Build a uniform strategy over every information set by running 0
    # iterations (all nodes default to uniform average strategy).
    strategy = leduc_cfr(iterations=0)
    # Even with no iterations, the average strategy is uniform over f/c/r.
    eps = exploitability(strategy)
    assert eps > 0.0, f"uniform strategy should be exploitable, got eps={eps}"


def test_cfr_converges_to_low_exploitability():
    """After enough iterations, CFR exploitability should be small.

    Vanilla CFR on Leduc typically reaches exploitability < 0.05 chips
    within 5000 iterations.  We use a generous threshold of 0.2 to
    keep the test fast (a few seconds) while still validating the
    convergence direction.
    """
    strategy = leduc_cfr(iterations=3000)
    eps = exploitability(strategy)
    assert eps < 0.2, f"CFR did not converge: exploitability={eps}"
    assert eps >= 0.0, "exploitability must be non-negative"


def test_exploitability_of_nash_is_zero():
    """Exploitability is non-negative and zero only at equilibrium.

    We check non-negativity on the CFR solution (which is near-Nash)
    and on a uniform strategy.  Exact zero is only at the true Nash.
    """
    strategy = leduc_cfr(iterations=2000)
    eps = exploitability(strategy)
    assert eps >= 0.0
    # The CFR solution should be strictly less exploitable than uniform.
    uniform = leduc_cfr(iterations=0)
    assert eps < exploitability(uniform)


# ---------------------------------------------------------------------------
# Best-response sanity
# ---------------------------------------------------------------------------


def test_best_response_values_match_published_game_value():
    """br0 and br1 bracket the game value; near-Nash it converges to it.

    The equilibrium value of standard Leduc (Southey et al., 2005
    betting: ante 1, raises 2/4, max 2 per round) is approximately
    -0.0856 for the first actor.  A near-Nash CFR profile must satisfy
    br0 >= v* and br1 >= -v*, so (br0 - br1)/2 -> v*.  Empirically
    vanilla CFR reaches |estimate - v*| < 0.03 within 1000 iterations.
    """
    strategy = leduc_cfr(iterations=1000)
    br0 = best_response_value(strategy, 0)
    br1 = best_response_value(strategy, 1)
    value_est = (br0 - br1) / 2
    assert abs(value_est - (-0.0856)) < 0.03, (
        f"game value estimate {value_est:.4f} far from published -0.0856; "
        f"br0={br0}, br1={br1}"
    )


def test_best_response_value_brackets_nash_value():
    """v* lies between -br1 and br0 (both sides gain from deviating)."""
    strategy = leduc_cfr(iterations=500)
    br0 = best_response_value(strategy, 0)
    br1 = best_response_value(strategy, 1)
    # v* in [-br1, br0] and the bracket is tight for near-Nash profiles.
    assert -br1 <= -0.0856 + 0.05, f"v* bound violated: br1={br1}"
    assert br0 >= -0.0856 - 0.05, f"v* bound violated: br0={br0}"


def test_average_strategy_is_valid_probability_distribution():
    strategy = leduc_cfr(iterations=500)
    assert len(strategy) > 0, "strategy table should not be empty"
    for info_set, probs in strategy.items():
        assert len(probs) == 3
        assert all(p >= 0.0 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6, f"{info_set}: sum={sum(probs)}"
