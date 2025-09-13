import random
import torch

from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer, ReplayBuffer


def test_perfect_reservoir_sampling():
    random.seed(0)
    buffer = ReplayBuffer(capacity=5)
    for i in range(10):
        s = torch.tensor([i], dtype=torch.float32)
        r = torch.tensor([i], dtype=torch.float32)
        buffer.push(s, r, i)
    iterations = [exp[2] for exp in buffer.buffer]
    assert sorted(iterations) == sorted([8, 1, 2, 5, 9])


def test_minibatch_shapes_masks():
    random.seed(0)
    torch.manual_seed(0)
    trainer = DeepCFRTrainer(input_feature_dim=4, hidden_dim=8, num_actions=2, buffer_capacity=50)
    for i in range(20):
        state = torch.ones(1, 4) * i
        regret = torch.tensor([float(i), -float(i)])
        trainer.replay_buffer.push(state, regret, i + 1)
    initial_loss = trainer.train(batch_size=5)
    for _ in range(5):
        loss = trainer.train(batch_size=5)
    assert loss <= initial_loss
    batch = trainer.replay_buffer.sample(5)
    states, regrets, iterations = zip(*batch, strict=False)
    assert torch.stack(states).shape == (5, 1, 4)
    assert torch.stack(regrets).shape == (5, 2)
    assert torch.tensor(iterations).shape == (5,)

