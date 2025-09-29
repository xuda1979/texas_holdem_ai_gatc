import types

import pytest
import torch

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.gui.playStrategy import ModelAIStrategy


@pytest.fixture(autouse=True)
def restore_helpers(monkeypatch):
    # Ensure utility functions are controlled per-test without leaking state.
    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.prepare_transformer_input",
        lambda *_args, **_kwargs: (
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.infer_normalization_scale",
        lambda *args, **kwargs: 1.0,
    )


def _make_model(num_actions: int, output: torch.Tensor) -> AdvantageNetwork:
    model = AdvantageNetwork(
        history_feature_dim=1,
        card_feature_dim=1,
        hidden_dim=1,
        num_heads=1,
        num_layers=1,
        num_actions=num_actions,
    )

    def _forward(self, *args, **kwargs):  # noqa: ANN001, ANN002 - test stub
        return output

    model.forward = types.MethodType(_forward, model)
    return model


def test_model_ai_strategy_prefers_best_legal_action(monkeypatch):
    torch.manual_seed(0)
    output = torch.tensor([3.0, 10.0, 2.0, 1.0])
    model = _make_model(4, output)

    mask = torch.tensor([True, False, True, False])
    captured = {}

    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.get_legal_actions_mask",
        lambda *_args, **_kwargs: mask,
    )
    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.get_action_from_index",
        lambda idx, *_args, **_kwargs: {0: "call", 2: "raise"}[idx],
    )
    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.action_to_tuple",
        lambda action: (action, None if action != "raise" else 50),
    )

    def _fake_multinomial(tensor, num_samples):  # noqa: ANN001
        captured["policy"] = tensor.clone()
        return torch.tensor([0])

    monkeypatch.setattr(torch, "multinomial", _fake_multinomial)

    strategy = ModelAIStrategy(model, {"num_actions": 4}, torch.device("cpu"))
    result = strategy.choose_action(object(), 0)

    assert result == ("call", None)
    # The illegal index (1) should be fully masked out despite having the largest raw advantage.
    assert captured["policy"][1].item() == 0.0
    assert torch.isclose(captured["policy"].sum(), torch.tensor(1.0))


def test_model_ai_strategy_samples_uniform_when_no_positive_advantages(monkeypatch):
    torch.manual_seed(0)
    output = torch.tensor([-1.0, 5.0, 0.0, -0.2])
    model = _make_model(4, output)

    mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    captured = {}

    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.get_legal_actions_mask",
        lambda *_args, **_kwargs: mask,
    )
    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.get_action_from_index",
        lambda idx, *_args, **_kwargs: {0: "check", 2: "raise"}[idx],
    )
    monkeypatch.setattr(
        "poker_ai.gui.playStrategy.action_to_tuple",
        lambda action: (action, None if action != "raise" else 25),
    )

    def _fake_multinomial(tensor, num_samples):  # noqa: ANN001
        captured["policy"] = tensor.clone()
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", _fake_multinomial)

    strategy = ModelAIStrategy(model, {"num_actions": 4}, torch.device("cpu"))
    result = strategy.choose_action(object(), 1)

    assert result == ("raise", 25)
    assert torch.allclose(captured["policy"], torch.tensor([0.5, 0.0, 0.5, 0.0]))


def test_model_ai_strategy_fallback_behaviour(monkeypatch):
    fallback_calls = []

    class Fallback:
        def choose_action(self, game, player_index):  # noqa: ANN001
            fallback_calls.append((game, player_index))
            return ("fold", None)

    strategy = ModelAIStrategy(
        model=None,
        config={"num_actions": 2},
        device=torch.device("cpu"),
        fallback_strategy=Fallback(),
    )

    outcome = strategy.choose_action("game", 3)
    assert outcome == ("fold", None)
    assert fallback_calls == [("game", 3)]

    # Simulate a misconfigured instance where neither a model nor fallback is available at runtime.
    strategy.model = None
    strategy._fallback_strategy = None
    with pytest.raises(RuntimeError):
        strategy.choose_action("game", 0)
