import hashlib
import logging
import os
import random

import torch
import torch.optim as optim

try:  # pragma: no cover - attempt to use PyYAML if available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML missing
    yaml = None
import torch.nn.functional as F

from poker_ai.ai.models.transformer import AdvantageNetwork

# CFR utilities live under the package namespace.
from poker_ai.rules.cfr import calculate_strategy, update_regret, update_strategy

# Load configuration.  If the YAML parser or file is missing we fall back to
# a small default config so that importing this module never fails.
try:
    if yaml is not None:
        with open(
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml"),
        ) as f:
            config = yaml.safe_load(f)  # type: ignore[arg-type]
    else:
        raise FileNotFoundError
except Exception:
    logging.warning(
        "config.yaml not found or PyYAML unavailable. Using default config values for AICFRTrainer.",
    )
    config = {
        "logging": {"log_file": "aicfr_trainer.log"},
        "model": {
            "hidden_dim": 128,
            "num_actions": 10,
            "learning_rate": 0.001,
            "d_raw_feature": 18,
            "max_seq_len": 256,
        },
        "training": {"save_model_path": "aicfr_model.pth"},
    }


# Setup logging
log_file_path = config["logging"]["log_file"]
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode="a")

# Expose config for package-level access so tests can override it
import sys

sys.modules[__package__ + ".config"] = config


class AICFRTrainer:
    def __init__(self, device: str | None = None):
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model_config = config.get("model", {})  # Get model sub-config, or empty dict
        hidden_dim = model_config.get("hidden_dim", 128)  # Default if not found
        output_dim = model_config.get("num_actions", 10)  # Default if not found
        learning_rate = model_config.get("learning_rate", 0.001)  # Default if not found

        # Feature dimensions for history sequence and card set summaries
        d_raw_feature = model_config.get("d_raw_feature", 18)
        d_card_feature = model_config.get("d_card_feature", 17)

        self.hidden_dim = hidden_dim
        self.num_heads = model_config.get("num_heads", 8)
        self.num_layers = model_config.get("num_layers", 2)

        self.model = AdvantageNetwork(
            history_feature_dim=d_raw_feature,
            card_feature_dim=d_card_feature,
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_actions=output_dim,
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_actions = output_dim  # Ensure this is consistent with model output
        self.history_feature_dim = d_raw_feature
        self.card_feature_dim = d_card_feature
        self.max_seq_len = model_config.get("max_seq_len", 256)
        # Expose configuration so callers (e.g. self-play) can retrieve model params
        self.config = config

        buffer_capacity = int(config.get("training", {}).get("replay_buffer_capacity", 100000))
        self.replay_buffer = AICFRReplayBuffer(buffer_capacity)

        # Track regrets and strategies per information set.
        # Keys are information set identifiers supplied during training.
        self.cumulative_regret: dict[str, torch.Tensor] = {}
        self.cumulative_strategy: dict[str, torch.Tensor] = {}

    def get_advantages(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the advantages for the given state representation."""

        if hole_summary.ndim == 1:
            hole_summary = hole_summary.unsqueeze(0)
        if community_summary.ndim == 1:
            community_summary = community_summary.unsqueeze(0)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)

        hole_summary = hole_summary.to(self.device)
        community_summary = community_summary.to(self.device)
        history_tensor = history_tensor.to(self.device)
        mask = mask.to(self.device) if mask is not None else None

        with torch.no_grad():
            advantages = self.model(hole_summary, community_summary, history_tensor, src_mask=None)
        return advantages.squeeze(0)

    def train(self, *args, **kwargs):
        """Dispatch training calls for compatibility with multiple front-ends."""

        if args and isinstance(args[0], str):
            return self._train_single(*args, **kwargs)

        batch_size = kwargs.get("batch_size")
        if batch_size is None and args:
            batch_size = args[0]
        if batch_size is None:
            batch_size = 256
        return self._train_from_buffer(int(batch_size))

    def _train_single(
        self,
        info_set_id: str,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        all_counterfactual_payoffs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> float:
        """Train the model for one infoset and return the loss."""

        try:
            if hole_summary.ndim == 1:
                hole_summary = hole_summary.unsqueeze(0)
            if community_summary.ndim == 1:
                community_summary = community_summary.unsqueeze(0)
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)

            hole_summary = hole_summary.to(self.device)
            community_summary = community_summary.to(self.device)
            history_tensor = history_tensor.to(self.device)
            payoffs = all_counterfactual_payoffs.to(self.device)

            logits = self.model(
                hole_summary,
                community_summary,
                history_tensor,
                src_mask=None,
            ).squeeze(0)
            strategy_pred = torch.softmax(logits, dim=-1)

            legal_mask = None
            if mask is not None:
                legal_mask = mask.to(self.device)
                if legal_mask.dtype != torch.bool:
                    legal_mask = legal_mask.bool()
                strategy_pred = torch.where(legal_mask, strategy_pred, torch.zeros_like(strategy_pred))
                prob_sum = strategy_pred.sum()
                if prob_sum.item() > 0:
                    strategy_pred = strategy_pred / prob_sum
                else:
                    legal_float = legal_mask.float()
                    total_legal = legal_float.sum()
                    if total_legal.item() > 0:
                        strategy_pred = legal_float / total_legal

                payoffs = torch.where(legal_mask, payoffs, torch.zeros_like(payoffs))

            if info_set_id not in self.cumulative_regret:
                self.cumulative_regret[info_set_id] = torch.zeros(
                    self.num_actions, device=self.device
                )
                self.cumulative_strategy[info_set_id] = torch.zeros(
                    self.num_actions, device=self.device
                )

            cumulative_regret = self.cumulative_regret[info_set_id]
            cumulative_strategy = self.cumulative_strategy[info_set_id]

            state_value = torch.sum(strategy_pred.detach() * payoffs)
            action_regrets = payoffs - state_value
            if legal_mask is not None:
                action_regrets = torch.where(legal_mask, action_regrets, torch.zeros_like(action_regrets))

            cumulative_regret = update_regret(cumulative_regret, action_regrets)
            current_regret_matched_policy = calculate_strategy(
                cumulative_regret, self.num_actions, legal_actions_mask=legal_mask
            )
            cumulative_strategy = update_strategy(
                cumulative_strategy, current_regret_matched_policy.detach()
            )

            self.cumulative_regret[info_set_id] = cumulative_regret
            self.cumulative_strategy[info_set_id] = cumulative_strategy

            loss = F.mse_loss(strategy_pred, current_regret_matched_policy.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logging.info(f"Training step completed. Loss: {loss.item()}")
            return float(loss.item())

        except Exception as e:  # pragma: no cover - logging path
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            raise

    def _train_from_buffer(self, batch_size: int) -> float:
        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return 0.0

        losses: list[float] = []
        for hole, community, history, payoffs, legal_mask, _iteration in batch:
            info_set_id = self._build_info_set_id(hole, community, history)
            loss = self._train_single(
                info_set_id,
                hole,
                community,
                history,
                payoffs,
                mask=legal_mask,
            )
            if loss is not None:
                losses.append(float(loss))
        if not losses:
            return 0.0
        return float(sum(losses) / len(losses))

    @staticmethod
    def _build_info_set_id(
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
    ) -> str:
        """Deterministically hash tensors into an information set identifier."""

        hasher = hashlib.sha1()
        for tensor in (hole_summary, community_summary, history_tensor):
            contiguous = tensor.detach().cpu().contiguous().view(-1)
            hasher.update(contiguous.numpy().tobytes())
        return hasher.hexdigest()

    def save_model(self, model_path=None):
        # Ensure config path is correct or make it an argument
        try:
            if model_path is None:
                model_path = self.config["training"]["save_model_path"]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
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
                    "trainer": "ai_cfr",
                },
            }
            torch.save(payload, model_path)
            logging.info(f"Model saved to {model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}", exc_info=True)

    def load_model(self):
        # Ensure config path is correct or make it an argument
        try:
            payload = torch.load(
                config["training"]["save_model_path"], map_location=self.device
            )
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded from {config['training']['save_model_path']}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}", exc_info=True)

    def get_final_average_strategy(self, info_set_id: str):
        """Return the average strategy for a given information set."""
        cumulative_strategy = self.cumulative_strategy.get(info_set_id)
        if cumulative_strategy is None:
            logging.warning(
                "Requested average strategy for unknown information set '%s'. Returning uniform.",
                info_set_id,
            )
            return torch.ones(self.num_actions, device=self.device) / self.num_actions

        sum_cumulative_strategy = torch.sum(cumulative_strategy)
        if sum_cumulative_strategy == 0:
            logging.warning(
                "Cumulative strategy is all zeros for information set '%s'. Returning uniform strategy.",
                info_set_id,
            )
            return torch.ones(self.num_actions, device=self.device) / self.num_actions
        return cumulative_strategy / sum_cumulative_strategy


class AICFRReplayBuffer:
    """Simple FIFO replay buffer for infoset experiences."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: list[tuple[torch.Tensor, ...]] = []

    def push(self, *args: torch.Tensor | int) -> None:
        if len(args) < 5:
            raise TypeError("push expects at least 5 arguments")

        hole, community, history, target, *remaining = args  # type: ignore[misc]
        iteration = int(remaining.pop()) if remaining else 0
        counterfactual_values = remaining.pop(0) if remaining else target
        legal_mask = remaining.pop(0) if remaining else None

        hole_cpu = hole.detach().cpu()
        community_cpu = community.detach().cpu()
        history_cpu = history.detach().cpu()
        cf_cpu = counterfactual_values.detach().cpu()
        if legal_mask is None:
            mask_cpu = torch.ones_like(cf_cpu, dtype=torch.bool)
        else:
            mask_cpu = legal_mask.detach().cpu().bool()

        entry = (
            hole_cpu,
            community_cpu,
            history_cpu,
            cf_cpu,
            mask_cpu,
            int(iteration),
        )

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
