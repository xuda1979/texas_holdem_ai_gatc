import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.rules.cfr import calculate_strategy, update_regret, update_strategy


class SingleNetworkCFRTrainer:
    """CFR trainer that predicts regret and strategy with a single network."""
    def __init__(self, input_feature_dim: int, hidden_dim: int, num_actions: int, lr: float = 1e-3,
                 device: str | None = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # ``AdvantageNetwork`` expects separate history and card feature dims; for
        # these lightweight tests we reuse ``input_feature_dim`` for both.
        self.model = AdvantageNetwork(
            history_feature_dim=input_feature_dim,
            card_feature_dim=input_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            num_actions=num_actions,
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_actions = num_actions
        self.cumulative_regret = torch.zeros(num_actions, device=self.device)
        self.cumulative_strategy = torch.zeros(num_actions, device=self.device)
        # Expose simple config for use by SelfPlay
        self.config = {
            'model': {
                'd_raw_feature': input_feature_dim,
                'hidden_dim': hidden_dim,
                'num_actions': num_actions,
                'learning_rate': lr,
            }
        }

    def train_step(self, state: torch.Tensor, counterfactual_payoffs: torch.Tensor):
        state = state.to(self.device)
        counterfactual_payoffs = counterfactual_payoffs.to(self.device)
        strategy_pred = self.model(state.unsqueeze(0)).squeeze(0)
        state_value = torch.sum(strategy_pred.detach() * counterfactual_payoffs)
        action_regrets = counterfactual_payoffs - state_value
        self.cumulative_regret = update_regret(self.cumulative_regret, action_regrets)
        target_strategy = calculate_strategy(self.cumulative_regret, self.num_actions)
        self.cumulative_strategy = update_strategy(self.cumulative_strategy, target_strategy.detach())
        loss = nn.functional.mse_loss(strategy_pred, target_strategy.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def average_strategy(self):
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return torch.ones(self.num_actions) / self.num_actions
