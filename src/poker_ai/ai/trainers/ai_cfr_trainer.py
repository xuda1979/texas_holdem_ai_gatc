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


def _load_xla_module():
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "TPU training requested but torch_xla is not installed. Install the torch-xla package."
        ) from exc
    return xm

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
    logging.getLogger(__name__).warning(
        "config.yaml not found or PyYAML unavailable. Using default config values for AICFRTrainer.",
    )
    config = {
        "logging": {"log_file": "aicfr_trainer.log"},
        "model": {
            "hidden_dim": 768,
            "num_actions": 10,
            "learning_rate": 0.001,
            "d_raw_feature": 18,
            "max_seq_len": 256,
            "num_layers": AdvantageNetwork.DEFAULT_NUM_LAYERS,
            "num_heads": AdvantageNetwork.DEFAULT_NUM_HEADS,
        },
        "training": {"save_model_path": "aicfr_model.pth"},
    }


# Expose config for package-level access so tests can override it
import sys

sys.modules[__package__ + ".config"] = config


class AICFRTrainer:
    def __init__(self, device: str | None = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
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
        self.logger.debug(
            "Initializing AICFRTrainer on device %s (using_xla=%s)",
            self.device,
            self._using_xla,
            extra={"component": "trainer"},
        )
        model_config = config.get("model", {})  # Get model sub-config, or empty dict
        hidden_dim = int(
            model_config.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM)
        )
        output_dim = int(model_config.get("num_actions", 10))
        learning_rate = float(model_config.get("learning_rate", 0.001))

        # Feature dimensions for history sequence and card set summaries
        d_raw_feature = model_config.get("d_raw_feature", 18)
        d_card_feature = model_config.get("d_card_feature", 17)

        self.hidden_dim = hidden_dim
        num_heads_config = model_config.get("num_heads")
        if num_heads_config is None:
            self.num_heads = AdvantageNetwork.recommended_num_heads(hidden_dim)
        else:
            self.num_heads = int(num_heads_config)
        self.num_layers = int(
            model_config.get("num_layers", AdvantageNetwork.DEFAULT_NUM_LAYERS)
        )

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

        target_device = self._xla_device or self.device
        hole_summary = hole_summary.to(target_device)
        community_summary = community_summary.to(target_device)
        history_tensor = history_tensor.to(target_device)
        mask = mask.to(target_device) if mask is not None else None

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
        opponent_reach: float = 1.0,
    ) -> float:
        """Train the model for one infoset and return the loss."""

        try:
            if hole_summary.ndim == 1:
                hole_summary = hole_summary.unsqueeze(0)
            if community_summary.ndim == 1:
                community_summary = community_summary.unsqueeze(0)
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)

            target_device = self._xla_device or self.device
            hole_summary = hole_summary.to(target_device)
            community_summary = community_summary.to(target_device)
            history_tensor = history_tensor.to(target_device)
            payoffs = all_counterfactual_payoffs.to(target_device)

            logits = self.model(
                hole_summary,
                community_summary,
                history_tensor,
                src_mask=None,
            ).squeeze(0)
            strategy_pred = torch.softmax(logits, dim=-1)

            legal_mask = None
            if mask is not None:
                legal_mask = mask.to(target_device)
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
                init_device = self._xla_device or self.device
                self.cumulative_regret[info_set_id] = torch.zeros(
                    self.num_actions, device=init_device
                )
                self.cumulative_strategy[info_set_id] = torch.zeros(
                    self.num_actions, device=init_device
                )

            cumulative_regret = self.cumulative_regret[info_set_id]
            cumulative_strategy = self.cumulative_strategy[info_set_id]

            state_value = torch.sum(strategy_pred.detach() * payoffs)
            action_regrets = payoffs - state_value
            if legal_mask is not None:
                action_regrets = torch.where(legal_mask, action_regrets, torch.zeros_like(action_regrets))
            reach_tensor = payoffs.new_tensor(float(opponent_reach))
            action_regrets = action_regrets * reach_tensor

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
            if self._using_xla:
                assert self._xm is not None
                self._xm.optimizer_step(self.optimizer)
                self._xm.mark_step()
            else:
                self.optimizer.step()

            self.logger.info(
                "Training step completed | loss=%.6f",
                float(loss.item()),
                extra={"component": "trainer"},
            )
            return float(loss.item())

        except Exception as e:  # pragma: no cover - logging path
            self.logger.exception("Error during training: %s", str(e))
            raise

    def _train_from_buffer(self, batch_size: int) -> float:
        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return 0.0

        losses: list[float] = []
        for entry in batch:
            opponent_reach = 1.0
            if len(entry) == 7:
                hole, community, history, payoffs, legal_mask, opponent_reach, _iteration = entry
            elif len(entry) == 6:
                hole, community, history, payoffs, legal_mask, _iteration = entry
            else:  # pragma: no cover - defensive guard for unexpected buffer format
                raise ValueError("Unexpected replay buffer entry format")
            info_set_id = self._build_info_set_id(hole, community, history)
            loss = self._train_single(
                info_set_id,
                hole,
                community,
                history,
                payoffs,
                mask=legal_mask,
                opponent_reach=float(opponent_reach),
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
            dirpath = os.path.dirname(model_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
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
            if self._using_xla:
                assert self._xm is not None
                self._xm.save(payload, model_path)
                self._xm.mark_step()
            else:
                torch.save(payload, model_path)
            self.logger.info(
                "Model saved | path=%s",
                model_path,
                extra={"component": "trainer"},
            )
        except Exception as e:
            self.logger.exception("Error saving model: %s", str(e))

    def load_model(self, model_path: str | None = None):
        # Ensure config path is correct or make it an argument
        try:
            training_cfg = self.config.get("training", {}) if isinstance(self.config, dict) else {}
            path = model_path or training_cfg.get("save_model_path") or config["training"]["save_model_path"]
            if not path:
                raise FileNotFoundError("No model path available to load weights.")
            map_location = self.device
            if self._using_xla:
                map_location = "cpu"
            payload = torch.load(path, map_location=map_location)
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            self.model.load_state_dict(state_dict)
            target_device = self._xla_device or self.device
            self.model.to(target_device)
            self.model.eval()
            self.logger.info(
                "Model loaded | path=%s",
                path,
                extra={"component": "trainer"},
            )
        except Exception as e:
            self.logger.exception("Error loading model: %s", str(e))

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

        target = action_values if action_values is not None else regrets
        if target is None:
            raise ValueError("AICFRTrainer.add_experience requires action_values or regrets.")
        self.replay_buffer.push(
            hole_summary,
            community_summary,
            history_tensor,
            target,
            legal_mask=legal_mask,
            iteration=iteration,
            opponent_reach=opponent_reach,
        )

    def get_final_average_strategy(self, info_set_id: str):
        """Return the average strategy for a given information set."""
        cumulative_strategy = self.cumulative_strategy.get(info_set_id)
        if cumulative_strategy is None:
            self.logger.warning(
                "Requested average strategy for unknown information set '%s'. Returning uniform.",
                info_set_id,
                extra={"component": "trainer"},
            )
            return torch.ones(self.num_actions, device=self.device) / self.num_actions

        sum_cumulative_strategy = torch.sum(cumulative_strategy)
        if sum_cumulative_strategy == 0:
            self.logger.warning(
                "Cumulative strategy is all zeros for information set '%s'. Returning uniform strategy.",
                info_set_id,
                extra={"component": "trainer"},
            )
            return torch.ones(self.num_actions, device=self.device) / self.num_actions
        return cumulative_strategy / sum_cumulative_strategy


class AICFRReplayBuffer:
    """Simple FIFO replay buffer for infoset experiences."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: list[tuple[torch.Tensor, ...]] = []

    def push(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        counterfactual_values: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        iteration: int = 0,
        *,
        opponent_reach: float | torch.Tensor = 1.0,
    ) -> None:
        hole_cpu = hole_summary.detach().cpu()
        community_cpu = community_summary.detach().cpu()
        history_cpu = history_tensor.detach().cpu()
        cf_cpu = counterfactual_values.detach().cpu()

        if legal_mask is None:
            mask_cpu = torch.ones_like(cf_cpu, dtype=torch.bool)
        else:
            mask_cpu = legal_mask.detach().cpu().bool()

        if torch.is_tensor(opponent_reach):
            reach_value = float(opponent_reach.detach().cpu().item())
        else:
            reach_value = float(opponent_reach)

        entry = (
            hole_cpu,
            community_cpu,
            history_cpu,
            cf_cpu,
            mask_cpu,
            reach_value,
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
