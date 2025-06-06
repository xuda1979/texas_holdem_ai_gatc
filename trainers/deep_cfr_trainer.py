import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple

from ai_models.transformer import TransformerAverageStrategy
from rules.cfr import calculate_strategy, update_regret, update_strategy

class ReplayBuffer:
    """Simple FIFO replay buffer for storing trajectories."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size: int):
        indices = torch.randperm(len(self.buffer))[:batch_size]
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class DeepCFRTrainer:
    """Minimal Deep CFR trainer using Transformer networks."""
    def __init__(self, input_feature_dim: int, hidden_dim: int, num_actions: int, learning_rate: float = 1e-3,
                 buffer_capacity: int = 10000):
        self.advantage_net = TransformerAverageStrategy(
            input_feature_dim=input_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            num_actions=num_actions,
        )
        self.strategy_net = TransformerAverageStrategy(
            input_feature_dim=input_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            num_actions=num_actions,
        )
        self.adv_optimizer = optim.Adam(self.advantage_net.parameters(), lr=learning_rate)
        self.strat_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.num_actions = num_actions
        self.cumulative_regret = torch.zeros(num_actions)
        self.cumulative_strategy = torch.zeros(num_actions)
        # Minimal config dict for compatibility with SelfPlay expectations
        self.config = {
            'model': {
                'd_raw_feature': input_feature_dim,
                'hidden_dim': hidden_dim,
                'num_actions': num_actions,
                'learning_rate': learning_rate,
            }
        }

    def store_trajectory(self, state: torch.Tensor, action: int, regret: torch.Tensor):
        self.replay_buffer.push((state.detach(), action, regret.detach()))

    def train_step(self, batch_size: int = 32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        regrets = torch.stack([b[2] for b in batch])

        # Train advantage network to predict regrets
        adv_pred = self.advantage_net(states)
        adv_loss = nn.functional.mse_loss(adv_pred, regrets)
        self.adv_optimizer.zero_grad()
        adv_loss.backward()
        self.adv_optimizer.step()

        # Update cumulative regret with predicted regrets for strategy training
        avg_regret = adv_pred.mean(dim=0)
        self.cumulative_regret = update_regret(self.cumulative_regret, avg_regret)
        strategy_target = calculate_strategy(self.cumulative_regret, self.num_actions)
        self.cumulative_strategy = update_strategy(self.cumulative_strategy, strategy_target.detach())

        # Train strategy network to match regret-matched policy
        strat_pred = self.strategy_net(states)
        strat_loss = nn.functional.mse_loss(strat_pred, strategy_target.expand_as(strat_pred).detach())
        self.strat_optimizer.zero_grad()
        strat_loss.backward()
        self.strat_optimizer.step()

    def get_average_strategy(self):
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return torch.ones(self.num_actions) / self.num_actions
