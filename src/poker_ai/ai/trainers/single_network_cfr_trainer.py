"""Single-network CFR trainer used by lightweight experiments.

The original implementation in this repository pre-dated the transformer based
``AdvantageNetwork`` and only consumed a flattened history tensor.  Recent
refactors switched the self-play pipeline to feed hole/board summaries as well
but the trainer was never updated, which meant ``run_training.py`` failed as
soon as the single-network algorithm was selected.  This module reintroduces a
fully functional trainer so the alternate algorithm is usable again.
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.rules.cfr import calculate_strategy, update_regret, update_strategy


def _load_xla_module():
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "TPU training requested but torch_xla is not installed. Install the torch-xla package."
        ) from exc
    return xm


class SingleNetworkCFRTrainer:
    """CFR trainer that uses one network to approximate action regrets."""

    def __init__(
        self,
        input_feature_dim: int,
        hidden_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
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

        # ``AdvantageNetwork`` expects separate history and card summaries.  The
        # simplified single-network approach reuses ``input_feature_dim`` for both
        # card projections which keeps compatibility with ``prepare_transformer_input``.
        self.history_feature_dim = input_feature_dim
        # Hole and community summaries are encoded with 17 features (13 ranks +
        # 4 suits) in :func:`prepare_transformer_input`.
        self.card_feature_dim = 17
        inferred_heads = AdvantageNetwork.recommended_num_heads(hidden_dim)
        self.model = AdvantageNetwork(
            history_feature_dim=self.history_feature_dim,
            card_feature_dim=self.card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=inferred_heads,
            num_layers=AdvantageNetwork.DEFAULT_NUM_LAYERS,
            num_actions=num_actions,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_heads = inferred_heads
        self.num_layers = AdvantageNetwork.DEFAULT_NUM_LAYERS
        self.max_seq_len = 256

        init_device = self._xla_device or self.device
        self.cumulative_regret = torch.zeros(num_actions, device=init_device)
        self.cumulative_strategy = torch.zeros(num_actions, device=init_device)

        # Minimal configuration dictionary consumed by ``SelfPlay`` and the
        # evaluation utilities.
        self.config = {
            "model": {
                "d_raw_feature": input_feature_dim,
                "d_card_feature": self.card_feature_dim,
                "hidden_dim": hidden_dim,
                "num_actions": num_actions,
                "learning_rate": lr,
                "max_seq_len": self.max_seq_len,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
            }
        }

        self.replay_buffer = _SingleNetworkReplayBuffer()

    @torch.no_grad()
    def get_advantages(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw advantage estimates for a given information set."""

        if hole_summary.ndim == 1:
            hole_summary = hole_summary.unsqueeze(0)
        if community_summary.ndim == 1:
            community_summary = community_summary.unsqueeze(0)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)

        target_device = self._xla_device or self.device
        logits = self.model(
            hole_summary.to(target_device),
            community_summary.to(target_device),
            history_tensor.to(target_device),
        ).squeeze(0)

        if mask is not None:
            legal_mask = mask.to(target_device)
            if legal_mask.dtype != torch.bool:
                legal_mask = legal_mask.bool()
            logits = torch.where(legal_mask, logits, torch.full_like(logits, -1e9))

        return logits.detach().cpu()

    def train_step(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        counterfactual_payoffs: torch.Tensor,
        legal_actions_mask: torch.Tensor | None = None,
    ) -> float:
        target_device = self._xla_device or self.device
        hole_summary = hole_summary.to(target_device)
        community_summary = community_summary.to(target_device)
        history_tensor = history_tensor.to(target_device)
        payoffs = counterfactual_payoffs.to(target_device)

        if hole_summary.ndim == 1:
            hole_summary = hole_summary.unsqueeze(0)
        if community_summary.ndim == 1:
            community_summary = community_summary.unsqueeze(0)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)

        logits = self.model(hole_summary, community_summary, history_tensor).squeeze(0)
        strategy_pred = torch.softmax(logits, dim=-1)

        mask = None
        if legal_actions_mask is not None:
            mask = legal_actions_mask.to(target_device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            strategy_pred = torch.where(mask, strategy_pred, torch.zeros_like(strategy_pred))
            prob_sum = strategy_pred.sum()
            if prob_sum.item() > 0:
                strategy_pred = strategy_pred / prob_sum
            else:
                mask_float = mask.float()
                total = mask_float.sum()
                if total.item() > 0:
                    strategy_pred = mask_float / total
            payoffs = torch.where(mask, payoffs, torch.zeros_like(payoffs))

        state_value = torch.sum(strategy_pred.detach() * payoffs)
        action_regrets = payoffs - state_value
        if mask is not None:
            action_regrets = torch.where(mask, action_regrets, torch.zeros_like(action_regrets))

        self.cumulative_regret = update_regret(self.cumulative_regret, action_regrets)
        target_strategy = calculate_strategy(
            self.cumulative_regret, self.num_actions, legal_actions_mask=mask
        )
        self.cumulative_strategy = update_strategy(
            self.cumulative_strategy, target_strategy.detach()
        )

        loss = nn.functional.mse_loss(strategy_pred, target_strategy.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self._using_xla:
            assert self._xm is not None
            self._xm.optimizer_step(self.optimizer)
            self._xm.mark_step()
        else:
            self.optimizer.step()
        return float(loss.item())

    def train(self, batch_size: int = 256) -> float:
        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return 0.0

        losses = [
            self.train_step(hole, community, history, payoffs, legal_actions_mask=mask)
            for hole, community, history, payoffs, mask in batch
        ]
        return float(sum(losses) / len(losses)) if losses else 0.0

    def average_strategy(self) -> torch.Tensor:
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return torch.ones(self.num_actions, device=self.device) / self.num_actions

    def save_model(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "metadata": {
                "history_feature_dim": self.history_feature_dim,
                "card_feature_dim": self.card_feature_dim,
                "num_actions": self.num_actions,
                "max_seq_len": self.max_seq_len,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "trainer": "single_network",
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
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        self.model.load_state_dict(state_dict)
        target_device = self._xla_device or self.device
        self.model.to(target_device)
        self.model.eval()


class _SingleNetworkReplayBuffer:
    """Replay buffer storing infoset tensors and counterfactual payoffs."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: list[tuple[torch.Tensor, ...]] = []

    def push(self, *args: torch.Tensor | int) -> None:
        if len(args) < 5:
            raise TypeError("push expects at least 5 arguments")

        hole, community, history, target, *remaining = args  # type: ignore[misc]
        _iteration = remaining.pop() if remaining else None
        counterfactual = remaining.pop(0) if remaining else target
        legal_mask = remaining.pop(0) if remaining else None

        hole_cpu = hole.detach().cpu()
        community_cpu = community.detach().cpu()
        history_cpu = history.detach().cpu()
        payoffs_cpu = counterfactual.detach().cpu()
        if legal_mask is None:
            mask_cpu = torch.ones_like(payoffs_cpu, dtype=torch.bool)
        else:
            mask_cpu = legal_mask.detach().cpu().bool()

        entry = (hole_cpu, community_cpu, history_cpu, payoffs_cpu, mask_cpu)

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(entry)

    def sample(self, batch_size: int) -> list[tuple[torch.Tensor, ...]]:
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, k)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)
