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


def _load_xla_module():
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:  # pragma: no cover - dependency missing in CPU/GPU environments
        raise RuntimeError(
            "TPU training requested but torch_xla is not installed. Install the torch-xla package."
        ) from exc
    return xm

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

        Accepts either ``(hole, community, history, regrets, iteration)``,
        ``(hole, community, history, regrets, counterfactual, legal_mask,
        iteration)`` from the modern self-play pipeline, or the legacy
        ``(state, regrets, iteration)`` tuple.
        """

        if len(args) == 3:
            history, regrets, iteration = args  # type: ignore[misc]
            hole = torch.zeros(self.card_feature_dim)
            community = torch.zeros(self.card_feature_dim)
        elif len(args) >= 5:
            hole, community, history, regrets = args[:4]  # type: ignore[misc]
            iteration = args[-1]
        else:  # pragma: no cover - defensive
            raise TypeError("push expects at least 3 arguments")

        if isinstance(iteration, torch.Tensor):
            if iteration.numel() != 1:  # pragma: no cover - defensive
                raise ValueError("iteration tensor must contain a single value")
            iteration = float(iteration.item())

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

        requested_device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._xm = None
        self._xla_device = None
        if isinstance(requested_device, torch.device) and requested_device.type == "xla":
            self._xm = _load_xla_module()
            self._xla_device = requested_device
        elif isinstance(requested_device, str) and requested_device.startswith("xla"):
            self._xm = _load_xla_module()
            ordinal: int | None = None
            try:
                _, ordinal_str = requested_device.split(":", 1)
                ordinal = int(ordinal_str)
            except (ValueError, IndexError):
                ordinal = None
            self._xla_device = (
                self._xm.xla_device(ordinal) if ordinal is not None else self._xm.xla_device()
            )
        if self._xla_device is not None:
            self.device = self._xla_device
        else:
            self.device = requested_device
        self._using_xla = self._xla_device is not None
        self.num_actions = num_actions

        # Each card summary is encoded as 17 features (13 rank + 4 suit).
        self.card_feature_dim = 17
        self.history_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = AdvantageNetwork.recommended_num_heads(hidden_dim)
        self.num_layers = AdvantageNetwork.DEFAULT_NUM_LAYERS
        self.max_seq_len = 256

        self.advantage_net = AdvantageNetwork(
            history_feature_dim=input_feature_dim,
            card_feature_dim=self.card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
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

        target_device = self._xla_device or self.device
        out = self.advantage_net(
            hole_summary.to(target_device),
            community_summary.to(target_device),
            history_tensor.to(target_device),
        )
        return out.squeeze(0).detach().cpu()

    def train(self, batch_size: int = 256) -> float:
        """Perform one training step using samples from the replay buffer."""

        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return 0.0

        holes, communities, histories, regrets, iterations = zip(*batch)

        target_device = self._xla_device or self.device
        holes = torch.stack(list(holes)).to(target_device)
        communities = torch.stack(list(communities)).to(target_device)
        histories = torch.stack(list(histories)).to(target_device)
        regrets = torch.stack(list(regrets)).to(target_device)
        iterations = torch.as_tensor(iterations, dtype=torch.float32, device=target_device).view(-1, 1)

        adv_pred = self.advantage_net(holes, communities, histories)
        adv_pred = adv_pred - adv_pred.mean(dim=-1, keepdim=True)
        regrets = regrets - regrets.mean(dim=-1, keepdim=True)

        loss_vals = (adv_pred - regrets) ** 2

        weights = iterations.clamp_min(0.0)
        weight_sum = weights.sum()
        if weight_sum.item() <= torch.finfo(loss_vals.dtype).eps:
            weighted_loss = loss_vals.mean()
        else:
            weighted_loss = (loss_vals * weights).sum() / weight_sum

        self.optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        clip_grad_norm_(self.advantage_net.parameters(), max_norm=1.0)
        if self._using_xla:
            assert self._xm is not None  # for type checkers
            self._xm.optimizer_step(self.optimizer)
            self._xm.mark_step()
        else:
            self.optimizer.step()

        return float(weighted_loss.item())

    def save_model(self, path: str) -> None:
        payload = {
            "state_dict": self.advantage_net.state_dict(),
            "metadata": {
                "history_feature_dim": self.history_feature_dim,
                "card_feature_dim": self.card_feature_dim,
                "num_actions": self.num_actions,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "max_seq_len": self.max_seq_len,
                "trainer": "deep_cfr",
            },
        }
        if self._using_xla:
            assert self._xm is not None
            self._xm.save(payload, path)
            self._xm.mark_step()
        else:
            torch.save(payload, path)

    def load_model(self, path: str) -> None:
        map_location = self.device
        if self._using_xla:
            map_location = "cpu"
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        self.advantage_net.load_state_dict(state_dict)
        target_device = self._xla_device or self.device
        self.advantage_net.to(target_device)
        self.advantage_net.eval()

