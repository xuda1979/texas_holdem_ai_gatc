"""Unit tests covering individual subsystem wrappers."""

from __future__ import annotations

from collections import Counter
import inspect

import pytest
import torch

from poker_ai.systems import (
    CFRSubsystem,
    EmbeddingSubsystem,
    EvaluationSubsystem,
    LoggingSubsystem,
    Registry,
    RulesSubsystem,
    SelfPlaySubsystem,
    Subsystem,
    TrainingSubsystem,
    TransformerSubsystem,
)


class DummyComponent:
    """Simple object used to exercise the base Subsystem helpers."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def ping(self, label: str) -> None:
        self.calls[label] += 1


def test_subsystem_describe_and_inject() -> None:
    component = DummyComponent()
    subsystem = Subsystem(name="dummy", component=component)
    subsystem.inject("alpha", object())
    subsystem.inject("beta", object())

    description = subsystem.describe()

    assert description == {
        "name": "dummy",
        "component": "DummyComponent",
        "dependencies": ["alpha", "beta"],
    }

    subsystem.component.ping("heartbeat")
    assert subsystem.component.calls["heartbeat"] == 1


def test_registry_tracks_unique_subsystems() -> None:
    registry = Registry()
    first = Subsystem(name="one", component=object())
    second = Subsystem(name="two", component=object())

    registry.register(first)
    registry.register(second)

    assert registry.get("one") is first
    assert registry.summary() == [
        {"name": "one", "component": "object", "dependencies": []},
        {"name": "two", "component": "object", "dependencies": []},
    ]

    with pytest.raises(ValueError):
        registry.register(Subsystem(name="one", component=object()))


def _advantage_tensors(component: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create tensors shaped for the supplied CFR trainer implementation."""

    card_dim = getattr(component, "card_feature_dim", 17)
    history_dim = getattr(component, "history_feature_dim", 18)
    num_actions = getattr(component, "num_actions", 10)

    hole = torch.zeros(card_dim)
    community = torch.zeros(card_dim)
    history = torch.zeros(history_dim)
    mask = torch.ones(num_actions)
    return hole, community, history, mask


@pytest.mark.parametrize(
    "variant, kwargs",
    [
        ("ai", {"device": "cpu"}),
        ("deep", {"input_feature_dim": 8, "hidden_dim": 16, "num_actions": 4, "device": "cpu"}),
        (
            "single",
            {"input_feature_dim": 6, "hidden_dim": 12, "num_actions": 5, "device": "cpu"},
        ),
    ],
)
def test_cfr_subsystem_variants_train_and_evaluate(variant: str, kwargs: dict[str, object]) -> None:
    subsystem = CFRSubsystem.create(variant=variant, **kwargs)

    loss = subsystem.train_from_buffer(batch_size=2)
    assert isinstance(loss, float)

    hole, community, history, mask = _advantage_tensors(subsystem.component)
    signature = inspect.signature(subsystem.component.get_advantages)
    extra_kwargs: dict[str, object] = {}
    if "mask" in signature.parameters:
        extra_kwargs["mask"] = mask
    captured: dict[str, object] = {}

    def fake_get_advantages(*args: object, **kwargs: object) -> torch.Tensor:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.ones(subsystem.component.num_actions)

    subsystem.component.get_advantages = fake_get_advantages  # type: ignore[assignment]

    advantages = subsystem.evaluate_state(hole, community, history, **extra_kwargs)

    assert advantages.shape[-1] == subsystem.component.num_actions
    assert captured["args"] == (hole, community, history)
    assert captured["kwargs"] == extra_kwargs


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_embedding_subsystem_builds_inputs(device: torch.device | str) -> None:
    embeddings = EmbeddingSubsystem.create(
        history_feature_dim=18,
        card_feature_dim=17,
        max_seq_len=8,
    )
    rules = RulesSubsystem.create(num_players=2, starting_stack=20, verbose=False)
    game = rules.new_game(verbose=False, cash_config={"enabled": False})
    game.initialize_game()

    hole, community, history, mask = embeddings.component.build_inputs(
        game,
        player_index=0,
        device=device,
        return_mask=True,
    )

    assert hole.device == torch.device("cpu")
    assert community.shape[-1] == embeddings.component.card_feature_dim
    assert history.shape[-1] == embeddings.component.history_feature_dim
    assert mask is not None and mask.dtype == torch.bool


def test_logging_subsystem_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    def _capture(name: str):
        def _wrapper(*args: object, **kwargs: object) -> None:
            recorded[name] = (args, kwargs)

        return _wrapper

    monkeypatch.setattr("poker_ai.logging_utils.setup_logging", _capture("setup"))
    monkeypatch.setattr("poker_ai.logging_utils.log_configuration_snapshot", _capture("snapshot"))
    monkeypatch.setattr("poker_ai.logging_utils.log_run_metadata", _capture("run_metadata"))

    subsystem = LoggingSubsystem.create(environment="unit-test")
    assert subsystem.dependencies == {"environment": "unit-test"}

    subsystem.setup(level="INFO")
    subsystem.snapshot({"seed": 42})
    subsystem.run_metadata(config={"iteration": 1})

    assert "setup" in recorded
    assert recorded["snapshot"][0] == ({"seed": 42},)
    assert recorded["run_metadata"][1] == {"config": {"iteration": 1}}


def test_transformer_subsystem_freeze_disables_gradients() -> None:
    subsystem = TransformerSubsystem.create(
        history_feature_dim=18,
        card_feature_dim=17,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        num_actions=6,
    )

    subsystem.freeze()

    for parameter in subsystem.component.parameters():
        assert parameter.requires_grad is False
    assert not subsystem.component.training


def test_self_play_subsystem_injects_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    cfr = CFRSubsystem.create(variant="ai", device="cpu")

    generated: list[int] = []

    def fake_play(iteration: int) -> list[str]:
        generated.append(iteration)
        return ["hand"]

    monkeypatch.setattr(
        "poker_ai.selfplay.self_play.SelfPlay.play_hand_for_training",
        lambda self, iteration: fake_play(iteration),
    )

    subsystem = SelfPlaySubsystem.create(
        cfr_trainer=cfr.component,
        game_engine_config={"starting_stack": 50, "min_players": 2, "max_players": 2},
        train_during_generation=False,
    )

    assert subsystem.dependencies["cfr_trainer"] is cfr.component

    result = subsystem.generate_training_hand(iteration=3)
    assert result == ["hand"]
    assert generated == [3]


def test_training_subsystem_coordinates_self_play(monkeypatch: pytest.MonkeyPatch) -> None:
    cfr = CFRSubsystem.create(variant="ai", device="cpu")

    play_iterations: list[int] = []

    def fake_generate(iteration: int) -> None:
        play_iterations.append(iteration)

    self_play = SelfPlaySubsystem.create(
        cfr_trainer=cfr.component,
        game_engine_config={"starting_stack": 10, "min_players": 2, "max_players": 2},
        train_during_generation=False,
    )
    self_play.component.play_hand_for_training = fake_generate

    training = TrainingSubsystem.create(cfr=cfr, self_play=self_play, iterations_per_cycle=3)

    losses: list[float] = []

    def fake_train(*, batch_size: int) -> float:
        losses.append(float(batch_size))
        return 1.5

    cfr.component.train = fake_train  # type: ignore[assignment]

    training.component.run_cycle(start_iteration=5)
    assert play_iterations == [5, 6, 7]

    warmup = list(training.component.warm_start(num_batches=2, batch_size=4))
    assert warmup == [1.5, 1.5]
    assert losses == [4.0, 4.0]


def test_subsystems_integration_round_trip() -> None:
    registry = Registry()

    cfr = CFRSubsystem.create(variant="ai", device="cpu")
    registry.register(cfr)

    rules = RulesSubsystem.create(num_players=2, starting_stack=30, verbose=False)
    registry.register(rules)

    embeddings = EmbeddingSubsystem.create(
        history_feature_dim=cfr.component.history_feature_dim,
        card_feature_dim=cfr.component.card_feature_dim,
        max_seq_len=cfr.component.max_seq_len,
    )
    registry.register(embeddings)

    transformer = TransformerSubsystem.create(
        history_feature_dim=cfr.component.history_feature_dim,
        card_feature_dim=cfr.component.card_feature_dim,
        hidden_dim=cfr.component.hidden_dim,
        num_heads=cfr.component.num_heads,
        num_layers=cfr.component.num_layers,
        num_actions=cfr.component.num_actions,
    )
    registry.register(transformer)

    logging_subsystem = LoggingSubsystem.create(component="integration-test")
    registry.register(logging_subsystem)

    evaluation = EvaluationSubsystem.create()

    def fake_tournament(*args: object, **kwargs: object) -> dict[str, int]:
        return {"bot": 3, "baseline": 1}

    evaluation.component.tournament_runner = fake_tournament
    registry.register(evaluation)

    self_play = SelfPlaySubsystem.create(
        cfr_trainer=cfr.component,
        game_engine_config={"starting_stack": 30, "min_players": 2, "max_players": 2},
        training_config={"train_during_generation": False},
        train_during_generation=False,
    )
    registry.register(self_play)

    self_play_calls: list[int] = []
    self_play.component.play_hand_for_training = lambda iteration: self_play_calls.append(iteration)

    training = TrainingSubsystem.create(cfr=cfr, self_play=self_play, iterations_per_cycle=2)
    registry.register(training)

    logging_subsystem.setup(level="WARNING")
    logging_subsystem.snapshot({"num_players": 2})
    logging_subsystem.run_metadata(config={"cycle": 0})

    game = rules.new_game(verbose=False, cash_config={"enabled": False})
    game.initialize_game()

    hole, community, history, mask = embeddings.component.build_inputs(game, player_index=0, return_mask=True)
    assert mask is not None

    advantage_record: dict[str, object] = {}

    def fake_advantages(*args: object, **kwargs: object) -> torch.Tensor:
        advantage_record["args"] = args
        advantage_record["kwargs"] = kwargs
        return torch.zeros(cfr.component.num_actions)

    cfr.component.get_advantages = fake_advantages  # type: ignore[assignment]

    advantages = cfr.evaluate_state(hole, community, history, mask=mask)
    assert advantages.shape[-1] == cfr.component.num_actions
    assert advantage_record["args"] == (hole, community, history)
    assert advantage_record["kwargs"] == {"mask": mask}

    list(training.component.warm_start(num_batches=1, batch_size=8))
    training.component.run_cycle(start_iteration=0)
    assert self_play_calls == [0, 1]

    standings = evaluation.component.run_tournament()
    assert standings == {"bot": 3, "baseline": 1}

    summary = registry.summary()
    assert {entry["name"] for entry in summary} == {
        "cfr:ai",
        "rules",
        "embeddings",
        "model:transformer",
        "logging",
        "evaluation",
        "self_play",
        "training",
    }
