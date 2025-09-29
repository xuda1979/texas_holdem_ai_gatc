import builtins

import pytest

from poker_ai.gui.playStrategy import HumanStrategy


class FakeRules:
    def __init__(self, pot, current_bet, bets, player_chips):
        self.pot = pot
        self.current_bet = current_bet
        self.bets = list(bets)
        self.player_chips = list(player_chips)


class FakeGame:
    def __init__(self, rules, valid_actions, min_raise, max_raise):
        self.rules = rules
        self._valid_actions = list(valid_actions)
        self._min_raise = min_raise
        self._max_raise = max_raise

    def get_valid_actions(self, player_index):
        return list(self._valid_actions)

    def get_min_raise_amount(self, player_index):
        return self._min_raise

    def get_max_raise_amount(self, player_index):
        return self._max_raise


@pytest.fixture(autouse=True)
def reset_display_cache(monkeypatch):
    # Ensure each test controls the optional analyzer hook explicitly.
    monkeypatch.setattr("poker_ai.evaluation.display_ai_gto_stats", None, raising=False)


def _set_inputs(monkeypatch, responses):
    iterator = iter(responses)

    def _fake_input(_prompt=""):
        try:
            return next(iterator)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise AssertionError("Input requested more times than expected") from exc

    monkeypatch.setattr(builtins, "input", _fake_input)


def test_choose_action_handles_raise_validation_and_optional_display(monkeypatch):
    calls: list[tuple[object, int]] = []
    monkeypatch.setattr(
        "poker_ai.evaluation.display_ai_gto_stats",
        lambda game, idx: calls.append((game, idx)),
        raising=False,
    )

    rules = FakeRules(pot=150, current_bet=20, bets=[0, 0], player_chips=[200, 150])
    game = FakeGame(rules, ["raise", "call", "fold"], min_raise=30, max_raise=80)
    _set_inputs(monkeypatch, ["raise", "abc", "25", "60"])

    action, amount = HumanStrategy().choose_action(game, 0)

    assert action == "raise"
    assert amount == 60
    assert calls == [(game, 0)]


def test_choose_action_returns_simple_actions_when_display_missing(monkeypatch):
    rules = FakeRules(pot=75, current_bet=15, bets=[15, 0], player_chips=[85, 120])
    game = FakeGame(rules, ["call", "fold"], min_raise=20, max_raise=40)
    _set_inputs(monkeypatch, ["call"])

    assert HumanStrategy().choose_action(game, 0) == ("call", None)


def test_choose_action_reprompts_when_raise_not_allowed(monkeypatch):
    rules = FakeRules(pot=90, current_bet=30, bets=[0, 0], player_chips=[40, 120])
    game = FakeGame(rules, ["raise", "call", "fold"], min_raise=50, max_raise=40)
    _set_inputs(monkeypatch, ["raise", "call"])

    assert HumanStrategy().choose_action(game, 0) == ("call", None)
