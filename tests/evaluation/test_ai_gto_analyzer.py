from __future__ import annotations

import traceback

import pytest

from poker_ai.evaluation import ai_gto_analyzer


@pytest.fixture(autouse=True)
def reset_analyzer_state():
    ai_gto_analyzer._loaded_cfr_trainer = None
    ai_gto_analyzer._trainer_config = {
        "num_actions": 4,
        "input_shape": (1, 1, 1),
        "learning_rate": 0.01,
    }
    ai_gto_analyzer._full_config = {
        "model": {"directory": "trained_models", "num_actions": 4},
        "training": {"save_model_path": "trained_models/cfr_model.pth"},
    }


class DummyModel:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


class DummyTrainer:
    init_calls = 0

    def __init__(self, config):  # noqa: ANN001 - signature dictated by analyzer
        DummyTrainer.init_calls += 1
        self.config = config
        self.model = DummyModel()
        self.load_calls = 0
        self.model_path = None

    def load_model(self, path):  # noqa: ANN001
        self.load_calls += 1
        self.model_path = path
        return True

    def encode_state(self, game):  # noqa: ANN001
        return "encoded"

    def get_strategy(self, state):  # noqa: ANN001
        return [0.1, 0.2, 0.3, 0.4]


def test_load_cfr_model_caches_trainer(monkeypatch):
    DummyTrainer.init_calls = 0
    monkeypatch.setattr(ai_gto_analyzer, "CFRTrainer", DummyTrainer)
    monkeypatch.setattr(ai_gto_analyzer.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai_gto_analyzer.os.path, "exists", lambda path: True)

    trainer1, config1 = ai_gto_analyzer.load_cfr_model_and_config("trained_models/cfr_model.pth")
    trainer2, config2 = ai_gto_analyzer.load_cfr_model_and_config("trained_models/cfr_model.pth")

    assert trainer1 is trainer2
    assert config1 == config2
    assert trainer1.model.eval_calls == 1
    assert trainer1.load_calls == 1
    assert DummyTrainer.init_calls == 1


def test_load_cfr_model_handles_missing_file(monkeypatch, capsys):
    monkeypatch.setattr(ai_gto_analyzer, "CFRTrainer", DummyTrainer)
    monkeypatch.setattr(ai_gto_analyzer.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai_gto_analyzer.os.path, "exists", lambda path: False)

    trainer, config = ai_gto_analyzer.load_cfr_model_and_config("trained_models/missing.pth")

    captured = capsys.readouterr().out
    assert "AI model file not found" in captured
    assert trainer is None
    assert config == ai_gto_analyzer._trainer_config


def test_display_ai_gto_stats_formats_probabilities(monkeypatch, capsys):
    trainer = DummyTrainer(ai_gto_analyzer._trainer_config)

    def _fake_loader(_path):
        return trainer, ai_gto_analyzer._trainer_config

    monkeypatch.setattr(ai_gto_analyzer, "load_cfr_model_and_config", _fake_loader)
    monkeypatch.setattr(ai_gto_analyzer, "action_to_tuple", lambda action: action)
    monkeypatch.setattr(
        ai_gto_analyzer,
        "get_action_from_index",
        lambda idx, *_args, **_kwargs: (
            ("fold", None),
            ("check", None),
            ("call", 20),
            ("raise", 60),
        )[idx],
    )

    class Rules:
        def __init__(self):
            self.pot = 150
            self.current_bet = 40
            self.bets = [10, 0]
            self.player_chips = [200, 300]

    class Game:
        def __init__(self):
            self.rules = Rules()

    ai_gto_analyzer.display_ai_gto_stats(Game(), 0)
    output = capsys.readouterr().out

    assert "Fold: 10.0%" in output
    assert "Check: 20.0%" in output
    assert "Call 30 chips" in output
    assert "Raise to 60" in output


def test_display_ai_gto_stats_logs_errors(monkeypatch, capsys):
    class ErrorTrainer(DummyTrainer):
        def get_strategy(self, state):  # noqa: ANN001
            raise RuntimeError("boom")

    trainer = ErrorTrainer(ai_gto_analyzer._trainer_config)

    def _fake_loader(_path):
        return trainer, ai_gto_analyzer._trainer_config

    monkeypatch.setattr(ai_gto_analyzer, "load_cfr_model_and_config", _fake_loader)
    monkeypatch.setattr(traceback, "print_exc", lambda: None)
    monkeypatch.setattr(ai_gto_analyzer, "action_to_tuple", lambda action: action)
    monkeypatch.setattr(
        ai_gto_analyzer,
        "get_action_from_index",
        lambda *_args, **_kwargs: ("fold", None),
    )

    class Game:
        def __init__(self):
            class Rules:
                pot = 0
                current_bet = 0
                bets = [0]
                player_chips = [0]

            self.rules = Rules()

    ai_gto_analyzer.display_ai_gto_stats(Game(), 0)
    output = capsys.readouterr().out
    assert "Error during AI GTO analysis" in output
