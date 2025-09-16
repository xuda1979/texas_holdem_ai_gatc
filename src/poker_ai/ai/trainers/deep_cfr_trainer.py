# ruff: noqa
"""Deep CFR trainer implementation with infoset replay buffer.

This module implements a lightweight version of the Deep Counterfactual
Regret Minimization (Deep CFR) algorithm using a transformer based advantage
network.  The trainer collects full information sets during self-play and
stores them in a reservoir-sampling replay buffer.  Training is performed with
the "Linear CFR" weighted mean-squared error loss.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from poker_ai.ai.models.transformer import AdvantageNetwork


class ReplayBuffer:
    """Reservoir-sampling replay buffer for Deep CFR."""

    def __init__(self, capacity: int, card_feature_dim: int = 17):
        self.capacity = capacity
        self.card_feature_dim = card_feature_dim
        self.buffer: list[Tuple[torch.Tensor, ...]] = []
        self.n_seen = 0  # total number of samples observed

    def push(self, *args: torch.Tensor | int) -> None:
        """Add an experience to the buffer using reservoir sampling.

        Accepts either ``(hole, community, history, regrets, iteration)`` or the
        legacy ``(state, regrets, iteration)`` tuple.
        """

        if len(args) == 5:
            hole, community, history, regrets, iteration = args  # type: ignore[misc]
        elif len(args) == 3:
            history, regrets, iteration = args  # type: ignore[misc]
            hole = torch.zeros(self.card_feature_dim)
            community = torch.zeros(self.card_feature_dim)
        else:  # pragma: no cover - defensive
            raise TypeError("push expects 5 or 3 arguments")

        exp = (
            hole.detach().cpu(),
            community.detach().cpu(),
            history.detach().cpu(),
            regrets.detach().cpu(),
            int(iteration),
        )
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            j = random.randrange(self.n_seen)
            if j < self.capacity:
                self.buffer[j] = exp

    def sample(self, batch_size: int) -> list[Tuple[torch.Tensor, ...]]:
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, k)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


class DeepCFRTrainer:
    """Deep CFR trainer using a transformer advantage network."""

    def __init__(
        self,
        input_feature_dim: int,
        hidden_dim: int,
        num_actions: int,
        learning_rate: float = 1e-4,
        replay_buffer_capacity: int = 1_000_000,
        buffer_capacity: int | None = None,
        device: str | None = None,
    ) -> None:

        if buffer_capacity is not None:
            replay_buffer_capacity = buffer_capacity

        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_actions = num_actions

        # Each card summary is encoded as 17 features (13 rank + 4 suit).
        self.card_feature_dim = 17

        self.advantage_net = AdvantageNetwork(
            history_feature_dim=input_feature_dim,
            card_feature_dim=self.card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            num_actions=num_actions,
        ).to(self.device)

        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity, self.card_feature_dim)

        # Minimal config dict retained for backward compatibility with callers.
        self.config = {
            "model": {
                "d_raw_feature": input_feature_dim,
                "hidden_dim": hidden_dim,
                "num_actions": num_actions,
                "learning_rate": learning_rate,
            }
        }

    @torch.no_grad()
    def get_advantages(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor | None = None,
        history_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return advantage estimates for the given infoset.

        Backward-compatible: older callers may pass a single ``history_tensor``
        without card summaries.  In that case zero summaries are used.
        """

        # Backward compatibility path: single tensor provided
        if history_tensor is None:
            history_tensor = hole_summary
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)
            batch = history_tensor.size(0)
            hole_summary = torch.zeros(batch, self.card_feature_dim, device=self.device)
            community_summary = torch.zeros(batch, self.card_feature_dim, device=self.device)
        else:
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)
            if hole_summary.ndim == 1:
                hole_summary = hole_summary.unsqueeze(0)
            if community_summary is not None and community_summary.ndim == 1:
                community_summary = community_summary.unsqueeze(0)

        out = self.advantage_net(
            hole_summary.to(self.device),
            community_summary.to(self.device),
            history_tensor.to(self.device),
        )
        return out.squeeze(0).detach().cpu()

    def train(self, batch_size: int = 256) -> float:
        """Perform one training step using samples from the replay buffer."""

        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return 0.0

        holes, communities, histories, regrets, iterations = zip(*batch)

        holes = torch.stack(list(holes)).to(self.device)
        communities = torch.stack(list(communities)).to(self.device)
        histories = torch.stack(list(histories)).to(self.device)
        regrets = torch.stack(list(regrets)).to(self.device)
        iterations = torch.as_tensor(iterations, dtype=torch.float32, device=self.device).view(-1, 1)

        adv_pred = self.advantage_net(holes, communities, histories)
        adv_pred = adv_pred - adv_pred.mean(dim=-1, keepdim=True)
        regrets = regrets - regrets.mean(dim=-1, keepdim=True)

        loss_vals = (adv_pred - regrets) ** 2
        weighted_loss = (loss_vals * iterations).sum() / iterations.sum()

        self.optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        clip_grad_norm_(self.advantage_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(weighted_loss.item())

    def save_model(self, path: str) -> None:
        torch.save(self.advantage_net.state_dict(), path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.advantage_net.load_state_dict(state)
        self.advantage_net.to(self.device)
        self.advantage_net.eval()

