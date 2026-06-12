from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.ai.trainers.deep_cfr_trainer import ReplayBuffer
from poker_ai.model_storage import prepare_model_write_path
from poker_ai.rules.cfr import calculate_strategy, update_regret, update_strategy


class SingleNetworkCFRTrainer:
    """Lightweight CFR trainer that relies on a single advantage network."""

    def __init__(
        self,
        input_feature_dim: int,
        hidden_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        device: str | None = None,
        buffer_capacity: int = 1_000_000,
    ) -> None:
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_actions = num_actions
        self.history_feature_dim = input_feature_dim
        self.card_feature_dim = 17
        self.hidden_dim = hidden_dim
        self.num_heads = AdvantageNetwork.recommended_num_heads(hidden_dim, preferred=4)
        self.num_layers = 2
        self.max_seq_len = 256

        self.model = AdvantageNetwork(
            history_feature_dim=self.history_feature_dim,
            card_feature_dim=self.card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_actions=num_actions,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.card_feature_dim)

        self.cumulative_regret = torch.zeros(num_actions, device=self.device)
        self.cumulative_strategy = torch.zeros(num_actions, device=self.device)

        self.config = {
            "model": {
                "d_raw_feature": self.history_feature_dim,
                "d_card_feature": self.card_feature_dim,
                "hidden_dim": hidden_dim,
                "num_actions": num_actions,
                "learning_rate": lr,
                "max_seq_len": self.max_seq_len,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
            }
        }

    @torch.no_grad()
    def get_advantages(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor | None = None,
        history_tensor: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return advantage estimates for the provided information set."""

        if history_tensor is None:
            history_tensor = hole_summary
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)
            batch = history_tensor.size(0)
            hole_summary = torch.zeros(batch, self.card_feature_dim, device=self.device)
            community_summary = torch.zeros(batch, self.card_feature_dim, device=self.device)
        else:
            if hole_summary.ndim == 1:
                hole_summary = hole_summary.unsqueeze(0)
            if community_summary is None:
                community_summary = torch.zeros_like(hole_summary)
            elif community_summary.ndim == 1:
                community_summary = community_summary.unsqueeze(0)
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)

        hole_summary = hole_summary.to(self.device)
        community_summary = community_summary.to(self.device)
        history_tensor = history_tensor.to(self.device)

        logits = self.model(hole_summary, community_summary, history_tensor)

        if mask is not None:
            legal_mask = mask.to(self.device)
            if legal_mask.dtype != torch.bool:
                legal_mask = legal_mask.bool()
            logits = torch.where(legal_mask, logits, torch.full_like(logits, -1e9))

        return logits.squeeze(0)

    @torch.no_grad()
    def _current_policy(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Regret-matched policy implied by the advantage network."""

        advantages = self.get_advantages(hole_summary, community_summary, history_tensor)
        positive = torch.clamp(advantages, min=0.0)
        if legal_mask is not None:
            mask = legal_mask.to(positive.device).bool()
            positive = torch.where(mask, positive, torch.zeros_like(positive))
        else:
            mask = None
        total = positive.sum()
        if total.item() > 0:
            return positive / total
        if mask is not None and bool(mask.any()):
            uniform = mask.to(positive.dtype)
            return uniform / uniform.sum()
        return torch.full_like(positive, 1.0 / positive.numel())

    def add_experience(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        *,
        action_values: torch.Tensor | None = None,
        regrets: torch.Tensor | None = None,
        legal_mask: torch.Tensor | None = None,
        opponent_reach: float | torch.Tensor = 1.0,
        iteration: int = 0,
    ) -> None:
        """Unified adapter for self-play experiences."""

        if regrets is None:
            if action_values is None:
                raise ValueError(
                    "SingleNetworkCFRTrainer.add_experience requires regrets or action_values."
                )
            values = action_values
            mask = None
            if legal_mask is not None:
                mask = legal_mask.to(values.device).bool()
                values = torch.where(mask, values, torch.zeros_like(values))
            # CFR instantaneous regret r(a) = Q(a) - V where the baseline is the
            # state value under the current regret-matched policy, not the
            # uniform mean of action values (which biases regret matching).
            sigma = self._current_policy(hole_summary, community_summary, history_tensor, mask)
            sigma = sigma.to(values.device, dtype=values.dtype)
            state_value = torch.sum(sigma * values)
            regrets = values - state_value
            if mask is not None:
                regrets = torch.where(mask, regrets, torch.zeros_like(regrets))
            if isinstance(opponent_reach, torch.Tensor):
                opponent_reach = float(opponent_reach.detach().cpu().item())
            regrets = regrets * float(opponent_reach)

        self.replay_buffer.push(
            hole_summary,
            community_summary,
            history_tensor,
            regrets,
            iteration,
        )

    def train(self, batch_size: int = 256) -> float:
        """Train the advantage network from experiences in the replay buffer."""

        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return 0.0

        holes, communities, histories, regrets, iterations = zip(*batch)
        holes = torch.stack(list(holes)).to(self.device)
        communities = torch.stack(list(communities)).to(self.device)
        histories = torch.stack(list(histories)).to(self.device)
        regrets = torch.stack(list(regrets)).to(self.device)
        iterations = torch.as_tensor(iterations, dtype=torch.float32, device=self.device).view(-1, 1)

        preds = self.model(holes, communities, histories)
        loss_vals = (preds - regrets) ** 2

        weights = iterations.clamp_min(0.0)
        weight_sum = weights.sum()
        if weight_sum.item() <= torch.finfo(loss_vals.dtype).eps:
            weighted_loss = loss_vals.mean()
        else:
            weighted_loss = (loss_vals * weights).sum() / weight_sum

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        return float(weighted_loss.item())

    def save_model(self, path: str) -> None:
        target = prepare_model_write_path(path)
        payload = {
            "state_dict": self.model.state_dict(),
            "metadata": {
                "history_feature_dim": self.history_feature_dim,
                "card_feature_dim": self.card_feature_dim,
                "num_actions": self.num_actions,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "max_seq_len": self.max_seq_len,
                "trainer": "single_network",
            },
        }
        torch.save(payload, str(target))

    def load_model(self, path: str) -> None:
        target = Path(path).expanduser()
        state = torch.load(str(target), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def train_step(self, state: torch.Tensor, counterfactual_payoffs: torch.Tensor) -> float:
        state = state.to(self.device)
        counterfactual_payoffs = counterfactual_payoffs.to(self.device)

        zeros = torch.zeros(1, self.card_feature_dim, device=self.device)
        logits = self.model(zeros, zeros, state.unsqueeze(0)).squeeze(0)
        strategy_pred = torch.softmax(logits, dim=-1)

        # Regrets are measured against sigma_t, the regret-matched policy from
        # the cumulative regrets prior to this update (standard CFR), rather
        # than the network's softmax approximation.
        sigma_t = calculate_strategy(self.cumulative_regret, self.num_actions).detach()
        state_value = torch.sum(sigma_t * counterfactual_payoffs)
        action_regrets = counterfactual_payoffs - state_value

        self.cumulative_regret = update_regret(self.cumulative_regret, action_regrets)
        target_strategy = calculate_strategy(self.cumulative_regret, self.num_actions)
        self.cumulative_strategy = update_strategy(
            self.cumulative_strategy, target_strategy.detach()
        )

        loss = nn.functional.mse_loss(strategy_pred, target_strategy.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def average_strategy(self) -> torch.Tensor:
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return torch.ones(self.num_actions, device=self.device) / self.num_actions
