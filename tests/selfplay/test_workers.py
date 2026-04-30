import pickle

import torch

from poker_ai.selfplay.distributed_self_play import DistributedSelfPlay, _hand_result_summary

NUM_HANDS = 4


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
    results = dsp.run(num_hands=NUM_HANDS, num_workers=2)
    assert len(results) == NUM_HANDS
    assert all(isinstance(r, list) for r in results)


def test_worker_result_summary_is_pickle_safe_for_tensor_buffers():
    replay_buffer = [(torch.zeros(2), torch.ones(2), 1)]

    summary = _hand_result_summary(replay_buffer)

    assert summary == ["replay_buffer_size", 1]
    pickle.dumps(summary)


def test_worker_result_summary_does_not_return_unpickleable_payload():
    class Unpickleable:
        __hash__ = None

        def __getstate__(self):
            raise RuntimeError("should not be pickled")

    summary = _hand_result_summary(Unpickleable())

    assert summary == ["result_type", "Unpickleable"]
    pickle.dumps(summary)
