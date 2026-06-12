import contextlib
import logging
from types import SimpleNamespace

from tools import run_diagnostics

TRAINING_LOSS = 0.5
TARGET_RETURN_CODE = 3
DIAGNOSTIC_MIN_BUFFER = 256


class TensorStub:
    def __init__(self, *shape):
        self.shape = tuple(shape)


class TorchStub:
    bool = "bool"

    @staticmethod
    def zeros(*shape, dtype=None):
        return TensorStub(*shape)

    @staticmethod
    def randn(*shape):
        return TensorStub(*shape)

    @staticmethod
    def ones(*shape, dtype=None):
        return TensorStub(*shape)

    @staticmethod
    @contextlib.contextmanager
    def no_grad():
        yield


class ProjectionStub:
    def __init__(self, num_features):
        self.in_features = num_features


class NetworkStub:
    def __init__(self, card_dim, history_dim, num_actions):
        self.card_projection = ProjectionStub(card_dim)
        self.history_projection = ProjectionStub(history_dim)
        self.num_actions = num_actions
        self.calls = []

    def __call__(self, hole, community, history):
        self.calls.append((hole.shape, community.shape, history.shape))
        return TensorStub(hole.shape[0], self.num_actions)


class TrainerStub:
    def __init__(self):
        self.model = NetworkStub(7, 11, 4)
        self.num_actions = 4
        self.card_feature_dim = 7
        self.history_feature_dim = 11
        self.max_seq_len = 9
        self.added = []
        self.batches = []

    def add_experience(self, hole, community, history, **kwargs):
        self.added.append((hole, community, history, kwargs))

    def train(self, batch_size=256):
        self.batches.append(batch_size)
        return 0.25


class AdvantageTrainerStub(TrainerStub):
    def __init__(self):
        super().__init__()
        self.advantage_net = NetworkStub(5, 13, 3)
        del self.model
        self.num_actions = 3
        self.card_feature_dim = 5
        self.history_feature_dim = 13
        self.max_seq_len = 6

    def train(self, batch_size=256):
        self.batches.append(batch_size)
        return TRAINING_LOSS


LOGGER = logging.getLogger("test.diagnostics")


def test_forward_pass_uses_model_when_available(monkeypatch):
    monkeypatch.setattr(run_diagnostics, "_load_torch", lambda: TorchStub)
    trainer = TrainerStub()

    result = run_diagnostics._diagnose_forward_pass(trainer, LOGGER)

    assert result["forward_pass_duration_s"] >= 0.0
    assert trainer.model.calls == [((1, 7), (1, 7), (1, 9, 11))]


def test_forward_pass_falls_back_to_advantage_network(monkeypatch):
    monkeypatch.setattr(run_diagnostics, "_load_torch", lambda: TorchStub)
    trainer = AdvantageTrainerStub()

    result = run_diagnostics._diagnose_forward_pass(trainer, LOGGER)

    assert result["forward_pass_duration_s"] >= 0.0
    assert trainer.advantage_net.calls == [((1, 5), (1, 5), (1, 6, 13))]


def test_training_step_uses_replay_buffer_interface(monkeypatch):
    monkeypatch.setattr(run_diagnostics, "_load_torch", lambda: TorchStub)
    trainer = AdvantageTrainerStub()

    result = run_diagnostics._diagnose_training_step(trainer, LOGGER)

    hole, community, history, kwargs = trainer.added[0]
    assert result["training_step_loss"] == TRAINING_LOSS
    assert result["training_step_duration_s"] >= 0.0
    assert trainer.batches == [1]
    assert hole.shape == (5,)
    assert community.shape == (5,)
    assert history.shape == (6, 13)
    assert kwargs["action_values"].shape == (3,)
    assert kwargs["legal_mask"].shape == (3,)
    assert kwargs["iteration"] == 1


def test_self_play_diagnostic_disables_training(monkeypatch):
    captured = {}

    class SelfPlayStub:
        min_buffer_before_train = 256

        def __init__(self, *, cfr_trainer, game_engine_config, training_config):
            captured["trainer"] = cfr_trainer
            captured["game_engine_config"] = game_engine_config
            captured["training_config"] = training_config

        def play_hand_for_training(self, iteration):
            captured["iteration"] = iteration
            return ["sample"]

    monkeypatch.setattr(run_diagnostics, "SelfPlay", SelfPlayStub)

    result = run_diagnostics._diagnose_self_play(
        trainer=object(),
        config={"training": {"min_buffer_before_train": 2, "train_during_generation": True}},
        logger=LOGGER,
    )

    assert result["replay_buffer_size"] == 1
    assert result["self_play_duration_s"] >= 0.0
    assert captured["training_config"]["train_during_generation"] is False
    assert captured["training_config"]["min_buffer_before_train"] == DIAGNOSTIC_MIN_BUFFER
    assert captured["iteration"] == 1


def test_build_iteration_plan_reports_impacted_subsystems():
    plan = run_diagnostics.build_iteration_plan(
        [
            "src/poker_ai/systems/training.py",
            "tests/selfplay/test_workers.py",
            "docs/common_workflows.md",
        ]
    )
    names = [item["name"] for item in plan["impacted_subsystems"]]

    assert "modular-services" in names
    assert "self-play-runtime" in names
    assert "training-orchestration" in names
    assert plan["unmatched_paths"] == ["docs/common_workflows.md"]
    assert "tests/selfplay" in plan["test_targets"]


def test_run_targeted_tests_invokes_pytest():
    calls = []

    def runner(command, cwd, check):
        calls.append((command, cwd, check))
        return SimpleNamespace(returncode=TARGET_RETURN_CODE)

    code = run_diagnostics.run_targeted_tests(["tests/test_simple.py"], runner=runner)

    assert code == TARGET_RETURN_CODE
    assert calls == [
        (
            [run_diagnostics.sys.executable, "-m", "pytest", "-q", "tests/test_simple.py"],
            str(run_diagnostics.ROOT),
            False,
        )
    ]
