"""Tests for opponent-pool self-play (fictitious-play style opponent mixing).

Verifies that ``SelfPlay`` honours the ``opponent_mix`` config: with the
feature enabled, a fraction of hands designate a non-traverser seat to play a
fixed baseline strategy (calling_station) instead of the model policy.
"""
from __future__ import annotations

import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest.mock import patch

from poker_ai.selfplay.self_play import SelfPlay


class DummyModel:
    num_actions = 10

    def __call__(self, x):
        return torch.zeros(self.num_actions)


class _Buffer(list):
    def push(self, *args, **kwargs):
        self.append(args)


class DummyTrainer:
    def __init__(self):
        self.model = DummyModel()
        self.config = {"model": {"max_seq_len": 10, "d_raw_feature": 2}}
        self.num_actions = 10
        self.replay_buffer = _Buffer()
        self.strategy_buffer = _Buffer()

    def get_advantages(self, hole, community, history):
        return torch.zeros(self.num_actions)

    def train(self, state_tensor=None, cf_payoffs=None, batch_size=None):
        return None

    def load_model(self, path=None):
        pass

    def add_strategy_experience(self, *args, **kwargs):
        pass

    def add_experience(self, *args, **kwargs):
        pass


def _make_sp(opponent_mix_probability=0.0, strategy="calling_station"):
    trainer = DummyTrainer()
    training_config = {
        "opponent_mix": {
            "probability": opponent_mix_probability,
            "strategy": strategy,
        },
        "min_buffer_before_train": 4,
    }
    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 50}, training_config)
    return sp


def test_opponent_mix_disabled_by_default():
    """With probability 0.0, no hand uses the fixed opponent."""
    sp = _make_sp(opponent_mix_probability=0.0)
    assert sp.opponent_mix_probability == 0.0
    assert sp._opponent_strategy is not None  # calling_station strategy loaded
    assert sp._current_fixed_opponent_id is None


def test_opponent_mix_config_parsed():
    """Config is parsed and clamped to [0, 1]."""
    sp = _make_sp(opponent_mix_probability=0.3)
    assert sp.opponent_mix_probability == 0.3
    assert sp.opponent_mix_strategy == "calling_station"
    assert sp._opponent_strategy is not None


def test_opponent_mix_probability_clamped():
    sp = _make_sp(opponent_mix_probability=5.0)
    assert sp.opponent_mix_probability == 1.0
    sp = _make_sp(opponent_mix_probability=-0.5)
    assert sp.opponent_mix_probability == 0.0


def test_unknown_strategy_disables_mix():
    """An unknown strategy name leaves the strategy object as None."""
    trainer = DummyTrainer()
    training_config = {
        "opponent_mix": {"probability": 0.5, "strategy": "nonexistent_strategy"},
    }
    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 50}, training_config)
    assert sp.opponent_mix_probability == 0.5
    assert sp._opponent_strategy is None


def test_flat_config_keys_also_supported():
    """The flat keys ``opponent_mix_probability`` / ``opponent_mix_strategy``
    are accepted as a fallback to the nested ``opponent_mix`` dict."""
    trainer = DummyTrainer()
    training_config = {
        "opponent_mix_probability": 0.2,
        "opponent_mix_strategy": "always_fold",
    }
    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 50}, training_config)
    assert sp.opponent_mix_probability == 0.2
    assert sp.opponent_mix_strategy == "always_fold"
    assert sp._opponent_strategy is not None


def test_fixed_opponent_action_index_returns_check_call_index():
    """The calling_station strategy maps to action index 1 (check/call) when
    that index is legal."""
    sp = _make_sp(opponent_mix_probability=0.5, strategy="calling_station")
    # 10-action mask with indices 0 and 1 legal.
    mask = torch.tensor([True, True, False, False, False, False, False, False, False, False])
    # Mock the game/strategy: calling_station returns ("check", None) or
    # ("call", None).  We patch the strategy object to return a known action.
    sp._opponent_strategy.choose_action = lambda game, pid: ("check", None)
    idx = sp._fixed_opponent_action_index(game=None, player_id=0, legal_actions_mask=mask)
    assert idx == 1


def test_fixed_opponent_action_index_returns_fold_when_legal():
    """The always_fold strategy maps to action index 0 when fold is legal."""
    sp = _make_sp(opponent_mix_probability=0.5, strategy="always_fold")
    mask = torch.tensor([True, True, False, False, False, False, False, False, False, False])
    sp._opponent_strategy.choose_action = lambda game, pid: ("fold", None)
    idx = sp._fixed_opponent_action_index(game=None, player_id=0, legal_actions_mask=mask)
    assert idx == 0


def test_fixed_opponent_action_index_returns_none_when_illegal():
    """If the strategy's preferred action index is illegal, return None so the
    caller falls back to the model policy."""
    sp = _make_sp(opponent_mix_probability=0.5, strategy="calling_station")
    # Index 1 illegal.
    mask = torch.tensor([True, False, False, False, False, False, False, False, False, False])
    sp._opponent_strategy.choose_action = lambda game, pid: ("check", None)
    idx = sp._fixed_opponent_action_index(game=None, player_id=0, legal_actions_mask=mask)
    assert idx is None


def test_play_hand_runs_with_opponent_mix_enabled():
    """A full ``play_hand_for_training`` call should not error with the
    opponent-mix feature enabled."""
    sp = _make_sp(opponent_mix_probability=1.0, strategy="calling_station")
    policy = torch.ones(sp.cfr_trainer.num_actions) / sp.cfr_trainer.num_actions
    mask = torch.ones(sp.cfr_trainer.num_actions, dtype=torch.bool)
    with (
        patch.object(SelfPlay, "_get_policy", return_value=(policy, mask)),
        patch(
            "poker_ai.selfplay.self_play.prepare_transformer_input",
            return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        ),
    ):
        data = sp.play_hand_for_training()
    assert isinstance(data, list)


def test_opponent_mix_engagement_counters():
    """The engagement counters track how many hands used the fixed opponent.

    With probability=1.0 every hand should be a mixed-opponent hand, so after
    N hands the counters should reflect N total and N mixed.
    """
    sp = _make_sp(opponent_mix_probability=1.0, strategy="calling_station")
    policy = torch.ones(sp.cfr_trainer.num_actions) / sp.cfr_trainer.num_actions
    mask = torch.ones(sp.cfr_trainer.num_actions, dtype=torch.bool)
    with (
        patch.object(SelfPlay, "_get_policy", return_value=(policy, mask)),
        patch(
            "poker_ai.selfplay.self_play.prepare_transformer_input",
            return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        ),
    ):
        for _ in range(5):
            sp.play_hand_for_training()
    assert sp._opponent_mix_total_hands == 5
    assert sp._opponent_mix_hands == 5


def test_opponent_mix_engagement_counters_disabled():
    """When the feature is disabled (probability=0), mixed hands stay at 0."""
    sp = _make_sp(opponent_mix_probability=0.0, strategy="calling_station")
    policy = torch.ones(sp.cfr_trainer.num_actions) / sp.cfr_trainer.num_actions
    mask = torch.ones(sp.cfr_trainer.num_actions, dtype=torch.bool)
    with (
        patch.object(SelfPlay, "_get_policy", return_value=(policy, mask)),
        patch(
            "poker_ai.selfplay.self_play.prepare_transformer_input",
            return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        ),
    ):
        for _ in range(3):
            sp.play_hand_for_training()
    assert sp._opponent_mix_total_hands == 3
    assert sp._opponent_mix_hands == 0
