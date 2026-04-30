from __future__ import annotations

# ruff: noqa
"""Deep CFR trainer implementation with infoset replay buffer.

This module implements a lightweight version of the Deep Counterfactual
Regret Minimization (Deep CFR) algorithm using a transformer based advantage
network.  The trainer collects full information sets during self-play and
stores them in a reservoir-sampling replay buffer.  Training is performed with
the "Linear CFR" weighted mean-squared error loss.
"""

import copy
import logging
import random
from pathlib import Path
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
from poker_ai.model_storage import prepare_model_write_path


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

        # Strategy Network (Policy Network) for Deep CFR
        self.policy_net = AdvantageNetwork(
            history_feature_dim=input_feature_dim,
            card_feature_dim=self.card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_actions=num_actions,
        ).to(self.device)

        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity, self.card_feature_dim)
        self.strategy_buffer = ReplayBuffer(replay_buffer_capacity, self.card_feature_dim)
        self._parallel_train_devices = self._resolve_parallel_train_devices()
        self._parallel_network_copies: dict[str, list[torch.nn.Module]] = {}
        self._parallel_training_logged = False

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

    def _resolve_parallel_train_devices(self) -> list[str]:
        """Return device strings to use for sharded training updates.

        The ai1 Ascend environment is stable when self-play inference remains on
        a single NPU, but it still exposes multiple visible NPUs that can be
        used safely for optimizer steps.  Keep the primary trainer model on the
        main device and shard only the training batch across the remaining NPUs.
        """

        if not isinstance(self.device, str) or not self.device.startswith("npu"):
            return []

        npu_module = getattr(torch, "npu", None)
        if npu_module is None:
            return []

        try:
            device_count = int(npu_module.device_count())
        except Exception:  # pragma: no cover - defensive
            return []

        if device_count <= 1:
            return []

        primary = self.device
        devices = [primary]
        devices.extend(f"npu:{idx}" for idx in range(1, device_count))
        return devices

    def _parallel_replicas(
        self, network_name: str, master_network: torch.nn.Module
    ) -> list[torch.nn.Module]:
        devices = self._parallel_train_devices
        if len(devices) <= 1:
            return [master_network]

        replicas = self._parallel_network_copies.get(network_name)
        if replicas is None or len(replicas) != len(devices):
            replicas = [master_network]
            for device in devices[1:]:
                replicas.append(copy.deepcopy(master_network).to(device))
            self._parallel_network_copies[network_name] = replicas

        state_dict = master_network.state_dict()
        replicas[0] = master_network
        for replica in replicas[1:]:
            replica.load_state_dict(state_dict)
        for replica in replicas:
            replica.train(master_network.training)
            replica.zero_grad(set_to_none=True)
        return replicas

    def _train_network_sharded(
        self,
        network_name: str,
        buffer: ReplayBuffer,
        optimizer: optim.Optimizer,
        batch_size: int,
        is_policy: bool,
    ) -> float:
        batch = buffer.sample(batch_size)
        if not batch:
            return 0.0

        master_network = getattr(self, network_name)
        devices = self._parallel_train_devices
        if len(devices) <= 1:
            return self._train_network(buffer, master_network, optimizer, batch_size, is_policy)

        holes, communities, histories, targets, iterations = zip(*batch)
        holes_cpu = torch.stack(list(holes))
        communities_cpu = torch.stack(list(communities))
        histories_cpu = torch.stack(list(histories))
        targets_cpu = torch.stack(list(targets))
        iterations_cpu = torch.as_tensor(iterations, dtype=torch.float32).view(-1, 1)

        chunk_count = min(len(devices), holes_cpu.size(0))
        if chunk_count <= 1:
            return self._train_network(buffer, master_network, optimizer, batch_size, is_policy)

        holes_chunks = list(torch.chunk(holes_cpu, chunk_count, dim=0))
        communities_chunks = list(torch.chunk(communities_cpu, chunk_count, dim=0))
        histories_chunks = list(torch.chunk(histories_cpu, chunk_count, dim=0))
        targets_chunks = list(torch.chunk(targets_cpu, chunk_count, dim=0))
        iteration_chunks = list(torch.chunk(iterations_cpu, chunk_count, dim=0))

        replicas = self._parallel_replicas(network_name, master_network)
        replicas = replicas[:chunk_count]
        optimizer.zero_grad(set_to_none=True)

        if not self._parallel_training_logged:
            logging.getLogger(__name__).info(
                "Using sharded Deep CFR training across %d devices for %s updates.",
                chunk_count,
                network_name,
            )
            self._parallel_training_logged = True

        total_weight = float(iterations_cpu.clamp_min(0.0).sum().item())
        total_elements = sum(int(chunk.numel()) for chunk in targets_chunks)
        use_weighted = total_weight > 0.0
        loss_value_numerator = 0.0

        for device, replica, hole_chunk, community_chunk, history_chunk, target_chunk, iter_chunk in zip(
            devices,
            replicas,
            holes_chunks,
            communities_chunks,
            histories_chunks,
            targets_chunks,
            iteration_chunks,
        ):
            hole_chunk = hole_chunk.to(device)
            community_chunk = community_chunk.to(device)
            history_chunk = history_chunk.to(device)
            target_chunk = target_chunk.to(device)
            iter_chunk = iter_chunk.to(device)

            preds = replica(hole_chunk, community_chunk, history_chunk)
            if not is_policy:
                preds = preds - preds.mean(dim=-1, keepdim=True)
                target_chunk = target_chunk - target_chunk.mean(dim=-1, keepdim=True)
                loss_vals = (preds - target_chunk) ** 2
            else:
                pred_probs = torch.softmax(preds, dim=-1)
                loss_vals = (pred_probs - target_chunk) ** 2

            if use_weighted:
                local_numerator = (loss_vals * iter_chunk.clamp_min(0.0)).sum()
                scaled_loss = local_numerator / total_weight
                loss_value_numerator += float(local_numerator.detach().cpu().item())
            else:
                local_numerator = loss_vals.sum()
                scaled_loss = local_numerator / max(total_elements, 1)
                loss_value_numerator += float(local_numerator.detach().cpu().item())

            scaled_loss.backward()

        master_params = list(master_network.parameters())
        for replica in replicas[1:]:
            for master_param, replica_param in zip(master_params, replica.parameters()):
                if replica_param.grad is None:
                    continue
                replica_grad = replica_param.grad.to(master_param.device)
                if master_param.grad is None:
                    master_param.grad = replica_grad
                else:
                    master_param.grad.add_(replica_grad)

        clip_grad_norm_(master_network.parameters(), max_norm=1.0)
        optimizer.step()

        if use_weighted:
            return loss_value_numerator / total_weight
        return loss_value_numerator / max(total_elements, 1)

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
        return out.squeeze(0)

    @torch.no_grad()
    def get_policy(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Return the strategy (probability distribution) from the policy network."""
        if hole_summary.ndim == 1:
            hole_summary = hole_summary.unsqueeze(0)
        if community_summary.ndim == 1:
            community_summary = community_summary.unsqueeze(0)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)

        target_device = self._xla_device or self.device
        logits = self.policy_net(
            hole_summary.to(target_device),
            community_summary.to(target_device),
            history_tensor.to(target_device),
        )
        return torch.softmax(logits, dim=-1).squeeze(0)

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
        """Unified adapter for self-play experiences (Advantage Memory)."""

        if regrets is None:
            if action_values is None:
                raise ValueError("DeepCFRTrainer.add_experience requires regrets or action_values.")
            processed = action_values
            if legal_mask is not None:
                legal_mask = legal_mask.to(processed.device).bool()
                processed = torch.where(legal_mask, processed, torch.zeros_like(processed))
            processed = processed - processed.mean()
            if isinstance(opponent_reach, torch.Tensor):
                opponent_reach = float(opponent_reach.detach().cpu().item())
            regrets = processed * float(opponent_reach)
        self.replay_buffer.push(
            hole_summary,
            community_summary,
            history_tensor,
            regrets,
            iteration,
        )

    def add_strategy_experience(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        strategy: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """Add a strategy sample to the Strategy Memory."""
        self.strategy_buffer.push(
            hole_summary,
            community_summary,
            history_tensor,
            strategy,
            iteration,
        )

    def train(self, batch_size: int = 256) -> float:
        """Perform one training step on the Advantage Network."""
        return self._train_network_sharded(
            "advantage_net",
            self.replay_buffer,
            self.optimizer,
            batch_size,
            is_policy=False,
        )

    def train_policy(self, batch_size: int = 256) -> float:
        """Perform one training step on the Policy Network."""
        return self._train_network_sharded(
            "policy_net",
            self.strategy_buffer,
            self.policy_optimizer,
            batch_size,
            is_policy=True,
        )

    def _train_network(
        self,
        buffer: ReplayBuffer,
        network: torch.nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int,
        is_policy: bool
    ) -> float:
        batch = buffer.sample(batch_size)
        if not batch:
            return 0.0

        holes, communities, histories, targets, iterations = zip(*batch)

        target_device = self._xla_device or self.device
        holes = torch.stack(list(holes)).to(target_device)
        communities = torch.stack(list(communities)).to(target_device)
        histories = torch.stack(list(histories)).to(target_device)
        targets = torch.stack(list(targets)).to(target_device)
        iterations = torch.as_tensor(iterations, dtype=torch.float32, device=target_device).view(-1, 1)

        pred = network(holes, communities, histories)

        if not is_policy:
            # Advantage network: Mean-centered MSE
            pred = pred - pred.mean(dim=-1, keepdim=True)
            targets = targets - targets.mean(dim=-1, keepdim=True)
            loss_vals = (pred - targets) ** 2
        else:
            # Policy network: Cross-Entropy or MSE on probabilities.
            # Deep CFR uses MSE on strategies usually, but CrossEntropy is also valid.
            # Here targets are probability distributions (from regret matching).
            # We use MSE as it matches the Deep CFR paper for strategy approximation.
            # pred output is raw logits. Apply softmax.
            pred_probs = torch.softmax(pred, dim=-1)
            loss_vals = (pred_probs - targets) ** 2

        weights = iterations.clamp_min(0.0)
        weight_sum = weights.sum()
        if weight_sum.item() <= torch.finfo(loss_vals.dtype).eps:
            weighted_loss = loss_vals.mean()
        else:
            weighted_loss = (loss_vals * weights).sum() / weight_sum

        optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        clip_grad_norm_(network.parameters(), max_norm=1.0)
        if self._using_xla:
            assert self._xm is not None  # for type checkers
            self._xm.optimizer_step(optimizer)
            self._xm.mark_step()
        else:
            optimizer.step()

        return float(weighted_loss.item())

    def save_model(self, path: str) -> None:
        target = prepare_model_write_path(path)
        payload = {
            "state_dict": self.advantage_net.state_dict(),
            "policy_net_state_dict": self.policy_net.state_dict(),
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
            self._xm.save(payload, str(target))
            self._xm.mark_step()
        else:
            torch.save(payload, str(target))

    def load_model(self, path: str) -> None:
        target = Path(path).expanduser()
        map_location = self.device
        if self._using_xla:
            map_location = "cpu"
        state = torch.load(str(target), map_location=map_location)
        if isinstance(state, dict):
            if "state_dict" in state:
                self.advantage_net.load_state_dict(state["state_dict"])
            if "policy_net_state_dict" in state:
                self.policy_net.load_state_dict(state["policy_net_state_dict"])
        else:
            # Fallback for old checkoints
            self.advantage_net.load_state_dict(state)

        target_device = self._xla_device or self.device
        self.advantage_net.to(target_device)
        self.advantage_net.eval()
        self.policy_net.to(target_device)
        self.policy_net.eval()
