from poker_ai.selfplay.distributed_self_play import DistributedSelfPlay

import torch


class DummyModel:
    num_actions = 2

    def __call__(self, x):
        return torch.zeros(self.num_actions)


class DummyBuffer(list):
    def push(self, *args):
        self.append(args)


class DummyTrainer:
    def __init__(self):
        self.model = DummyModel()
        self.config = {"model": {"max_seq_len": 10, "d_raw_feature": 2}}
        self.num_actions = 2
        self.replay_buffer = DummyBuffer()

    def get_advantages(self, hole, community, history):
        return torch.zeros(self.num_actions)

    def train(self, batch_size=None):
        return None


def test_spawn_n_workers_and_collect_k_games():
    trainer = DummyTrainer()
    dsp = DistributedSelfPlay(trainer, {"num_players": 2, "starting_stack": 50})
    results = dsp.run(num_hands=4, num_workers=2)
    assert len(results) == 4
    assert all(isinstance(r, list) for r in results)
