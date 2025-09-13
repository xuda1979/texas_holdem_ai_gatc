from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

if sys.version_info >= (3, 12):
    pytest.skip("Hypothesis providers are incompatible with Python 3.12", allow_module_level=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (project_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from gatc_holdem.engine.rules import min_raise_to, raise_reopens_action  # noqa: E402
from poker_ai.utils.action_mapping import _is_action_valid  # noqa: E402


@st.composite
def game_states(draw: DrawFn) -> tuple[SimpleNamespace, int, str, int | None]:
    num_players = draw(st.integers(min_value=2, max_value=6))
    player_id = draw(st.integers(min_value=0, max_value=num_players - 1))
    big_blind = draw(st.integers(min_value=1, max_value=1000))
    current_bet = draw(st.integers(min_value=0, max_value=2000))
    bets = draw(
        st.lists(
            st.integers(min_value=0, max_value=current_bet),
            min_size=num_players,
            max_size=num_players,
        )
    )
    player_chips = draw(
        st.lists(
            st.integers(min_value=0, max_value=20000),
            min_size=num_players,
            max_size=num_players,
        )
    )
    prev_raise = draw(st.integers(min_value=0, max_value=2000))
    game = SimpleNamespace(
        rules=SimpleNamespace(
            big_blind=big_blind,
            current_bet=current_bet,
            previous_raise_amount=prev_raise,
            player_chips=player_chips,
            bets=bets,
        )
    )
    action = draw(st.sampled_from(["fold", "check", "call", "bet", "raise"]))
    amount = draw(st.integers(min_value=0, max_value=40000)) if action in {"bet", "raise"} else None
    return game, player_id, action, amount


@given(game_states())
def test_is_action_valid_matches_naive(
    game_states: tuple[SimpleNamespace, int, str, int | None],
) -> None:
    """Compare engine's action legality check with a naive implementation."""
    game, pid, action, amount = game_states
    result = _is_action_valid(game, pid, action, amount)
    rules = game.rules
    player_bet = rules.bets[pid]
    player_stack = rules.player_chips[pid]
    amount_to_call = rules.current_bet - player_bet

    min_raise_inc = (
        rules.previous_raise_amount if rules.previous_raise_amount > 0 else rules.big_blind
    )

    if action == "fold":
        expected = True
    elif action == "check":
        expected = amount_to_call == 0
    elif action == "call":
        expected = amount_to_call > 0
    elif action == "bet":
        expected = (
            rules.current_bet == 0
            and amount is not None
            and amount > 0
            and (amount >= rules.big_blind or amount == player_stack)
            and player_stack >= amount
        )
    elif action == "raise":
        if rules.current_bet == 0 or amount is None or amount <= rules.current_bet:
            expected = False
        else:
            raise_commit = amount - player_bet
            actual_inc = amount - rules.current_bet
            if player_stack < raise_commit:
                expected = False
            elif actual_inc < min_raise_inc and raise_commit != player_stack:
                expected = False
            else:
                expected = True
    else:
        expected = False

    assert result == expected


@given(
    current_bet=st.integers(min_value=0, max_value=10000),
    last_raise=st.integers(min_value=0, max_value=10000),
    extra=st.integers(min_value=0, max_value=10000),
)
def test_raise_reopens_action_equivalence(current_bet: int, last_raise: int, extra: int) -> None:
    """Ensure reopen logic matches whether the extra meets the last raise."""
    raise_to = current_bet + extra
    assert raise_reopens_action(raise_to, current_bet, last_raise) == (extra >= last_raise)


@given(
    current_bet=st.integers(min_value=0, max_value=10000),
    last_raise=st.integers(min_value=0, max_value=10000),
    big_blind=st.integers(min_value=1, max_value=10000),
)
def test_min_raise_to_formula(current_bet: int, last_raise: int, big_blind: int) -> None:
    """Check minimum raise formula against rule helper."""
    result = min_raise_to(current_bet, last_raise, big_blind)
    if current_bet == 0:
        assert result == big_blind
    else:
        assert result == current_bet + max(last_raise, 0)


@given(
    current_bet=st.integers(min_value=0, max_value=10000),
    last_raise=st.integers(min_value=0, max_value=10000),
    deficit=st.integers(min_value=1, max_value=10000),
)
def test_raise_reopens_action_invalid(current_bet: int, last_raise: int, deficit: int) -> None:
    """Raises lower than the current bet should error."""
    with pytest.raises(ValueError):
        raise_reopens_action(current_bet - deficit, current_bet, last_raise)
