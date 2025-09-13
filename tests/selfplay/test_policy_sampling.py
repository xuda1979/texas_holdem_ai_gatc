import logging
import random
import torch

from poker_ai.selfplay.self_play import SelfPlay
from poker_ai.engine.texas_holdem import TexasHoldem


class DummyModel:
    num_actions = 4

    def __call__(self, x):
        return torch.zeros(self.num_actions)


class DummyBuffer(list):
    def push(self, *args):
        self.append(args)


class DummyTrainer:
    def __init__(self):
        self.model = DummyModel()
        self.config = {"model": {"max_seq_len": 10, "d_raw_feature": 2}}
        self.num_actions = 4
        self.replay_buffer = DummyBuffer()

    def get_advantages(self, state_tensor):
        # return zero advantages for deterministic uniform policy
        return torch.zeros(self.num_actions)

    def train(self, batch_size=None):
        return None


def _make_game():
    game = TexasHoldem(num_players=2, starting_stack=50, verbose=True)
    game.initialize_game()
    return game


def test_action_probs_sum_to_one():
    trainer = DummyTrainer()
    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 50})
    game = _make_game()
    policy = sp._get_policy(game, player_id=0)
    assert torch.isclose(policy.sum(), torch.tensor(1.0))
    # ensure only legal actions have positive probability
    legal_mask = policy > 0
    assert policy[~legal_mask].sum() == 0


def test_private_cards_hidden_in_logs(caplog):
    with caplog.at_level(logging.INFO):
        game = _make_game()
    # format each player's hole cards
    for hand in game.rules.hands:
        for card in hand:
            formatted = game.rules._format_card(card)
            assert formatted not in caplog.text
    assert "Player 1's hand" not in caplog.text


def test_seed_controlled_replay_equality():
    game_cfg = {"num_players": 2, "starting_stack": 50}
    seed = 123

    random.seed(seed)
    torch.manual_seed(seed)
    trainer1 = DummyTrainer()
    sp1 = SelfPlay(trainer1, game_cfg)
    data1 = sp1.play_hand_for_training(iteration=0).copy()

    random.seed(seed)
    torch.manual_seed(seed)
    trainer2 = DummyTrainer()
    sp2 = SelfPlay(trainer2, game_cfg)
    data2 = sp2.play_hand_for_training(iteration=0).copy()

    assert len(data1) == len(data2)
    for a, b in zip(data1, data2):
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
        assert a[2] == b[2]
