import torch
from self_play.self_play import SelfPlay


class DummyModel:
    num_actions = 2

    def __call__(self, x):
        return torch.zeros(self.num_actions)


class DummyTrainer:
    def __init__(self):
        self.model = DummyModel()

    def train(self, state_tensor, cf_payoffs):
        pass


def test_play_hand_for_training_runs():
    trainer = DummyTrainer()
    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 50})
    data = sp.play_hand_for_training()
    assert isinstance(data, list)
