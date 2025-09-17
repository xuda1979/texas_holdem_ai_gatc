"""
A command-line interface to play against a trained Poker AI model.
"""

import argparse
import inspect
import os
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

# ruff: noqa: ANN201,ANN204
from poker_ai.ai.model_loader import load_model_strategy
from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import HumanStrategy, ModelAIStrategy, PlayerStrategy
from poker_ai.rules.cfr import calculate_strategy
from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import (
    infer_normalization_scale,
    prepare_transformer_input,
)


class AIStrategy(PlayerStrategy):
    """A strategy that uses a trained AdvantageNetwork to make decisions."""

    @staticmethod
    def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if not any(key.startswith("module.") for key in state_dict):
            return state_dict
        return OrderedDict(
            (key.partition(".")[2] if key.startswith("module.") else key, value)
            for key, value in state_dict.items()
        )

    @staticmethod
    def _infer_metadata_from_state(
        state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any] | None = None
    ) -> dict[str, int] | None:
        base: dict[str, Any] = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    base[key] = int(value)

        def _get_tensor(suffix: str) -> torch.Tensor | None:
            for key, tensor in state_dict.items():
                if key.endswith(suffix) and isinstance(tensor, torch.Tensor):
                    return tensor
            return None

        history_weight = _get_tensor("history_projection.weight")
        card_weight = _get_tensor("card_projection.weight")
        fc_weight = _get_tensor("fc.weight")

        if history_weight is None or history_weight.ndim != 2:
            return None
        if card_weight is None or card_weight.ndim != 2:
            return None
        if fc_weight is None or fc_weight.ndim != 2:
            return None

        hidden_dim = int(history_weight.shape[0])
        history_dim = int(history_weight.shape[1])
        card_dim = int(card_weight.shape[1])
        num_actions = int(fc_weight.shape[0])

        base.setdefault("hidden_dim", hidden_dim)
        base.setdefault("history_feature_dim", history_dim)
        base.setdefault("card_feature_dim", card_dim)
        base.setdefault("num_actions", num_actions)

        if base["hidden_dim"] <= 0 or base["num_actions"] <= 0:
            return None

        if "hidden_dim" not in base and fc_weight.shape[1] % 3 == 0:
            base["hidden_dim"] = int(fc_weight.shape[1] // 3)

        layer_prefix = "transformer.layers."
        layer_indices = {
            int(parts[0])
            for key in state_dict
            if key.startswith(layer_prefix)
            and (parts := key[len(layer_prefix) :].split("."))
            and parts[0].isdigit()
        }
        if layer_indices:
            base.setdefault("num_layers", len(layer_indices))

        if "num_layers" not in base:
            base["num_layers"] = 2

        if "num_heads" not in base:
            for candidate in (8, 6, 5, 4, 3, 2):
                if base["hidden_dim"] % candidate == 0:
                    base["num_heads"] = candidate
                    break
            base.setdefault("num_heads", 1)

        base.setdefault("max_seq_len", 256)
        base.setdefault("d_raw_feature", base["history_feature_dim"])
        base.setdefault("input_feature_dim", base["history_feature_dim"])

        try:
            return {key: int(value) for key, value in base.items()}
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    @classmethod
    def _recover_model_from_path(
        cls, model_path: str, map_location: torch.device | str
    ) -> tuple[AdvantageNetwork, dict[str, int]] | None:
        try:
            payload = torch.load(model_path, map_location=map_location)
        except Exception:
            return None

        state_dict: Mapping[str, torch.Tensor] | None = None
        metadata: Mapping[str, Any] | None = None

        if isinstance(payload, Mapping):
            candidate = payload.get("state_dict")
            metadata_candidate = payload.get("metadata")
            if isinstance(candidate, Mapping):
                state_dict = candidate
            elif all(isinstance(v, torch.Tensor) for v in payload.values()):
                state_dict = payload  # type: ignore[assignment]
            if isinstance(metadata_candidate, Mapping):
                metadata = metadata_candidate
        elif isinstance(payload, OrderedDict):  # pragma: no cover - handled above
            state_dict = payload

        if state_dict is None:
            return None

        stripped = cls._strip_module_prefix(state_dict)
        metadata_int = cls._infer_metadata_from_state(stripped, metadata)
        if metadata_int is None:
            return None

        try:
            model = AdvantageNetwork(
                history_feature_dim=metadata_int["history_feature_dim"],
                card_feature_dim=metadata_int["card_feature_dim"],
                hidden_dim=metadata_int["hidden_dim"],
                num_heads=metadata_int["num_heads"],
                num_layers=metadata_int["num_layers"],
                num_actions=metadata_int["num_actions"],
            )
            model.load_state_dict(stripped)
        except Exception:
            return None

        return model, metadata_int

    def __init__(
        self,
        model_path: str,
        device: str,
        num_actions: int = 10,
        use_all_npus: bool = False,
    ):
        self.device = device
        self.num_actions = num_actions
        self.model = None
        self.config: dict[str, object] = {}
        self._fallback_strategy: PlayerStrategy | None = None
        self.max_seq_len = 256
        self.feature_dim = 18

        loaded_strategy, loader_device = load_model_strategy(model_path)

        try:
            target_device = torch.device(device)
        except (TypeError, ValueError, RuntimeError):
            target_device = loader_device

        self._torch_device = target_device

        wrap_for_npu = (
            use_all_npus
            and (device == "npu" or getattr(target_device, "type", None) == "npu")
            and hasattr(torch, "npu")
            and torch.npu.is_available()
            and torch.npu.device_count() > 1
        )

        if isinstance(loaded_strategy, ModelAIStrategy):
            model = loaded_strategy.model.to(target_device)

            if wrap_for_npu:
                model = torch.nn.DataParallel(model)

            self.model = model
            self.model.eval()

            self.config = dict(loaded_strategy.config)
            self.max_seq_len = int(self.config.get("max_seq_len", self.max_seq_len))
            feature_dim = self.config.get(
                "d_raw_feature",
                self.config.get(
                    "input_feature_dim",
                    self.config.get("history_feature_dim", 18),
                ),
            )
            self.feature_dim = int(feature_dim)
            configured_num_actions = int(
                self.config.get(
                    "num_actions",
                    getattr(getattr(self.model, "module", self.model), "num_actions", num_actions),
                )
            )
            self.num_actions = configured_num_actions
            self.config["num_actions"] = self.num_actions
        else:
            recovered = self._recover_model_from_path(model_path, loader_device)
            if recovered is not None:
                model, metadata = recovered
                model = model.to(target_device)
                if wrap_for_npu:
                    model = torch.nn.DataParallel(model)
                self.model = model
                self.model.eval()
                self.config = dict(metadata)
                self.max_seq_len = int(self.config.get("max_seq_len", self.max_seq_len))
                self.feature_dim = int(self.config.get("d_raw_feature", self.feature_dim))
                self.num_actions = int(self.config.get("num_actions", self.num_actions))
                self.config["num_actions"] = self.num_actions
                self._fallback_strategy = None
            else:
                self._fallback_strategy = loaded_strategy

    @property
    def is_human(self):
        if self._fallback_strategy is not None:
            return self._fallback_strategy.is_human
        return False

    def _normalization_scale(self, game: TexasHoldem) -> float:
        preferred_scale = None
        for key in ("normalization_scale", "chip_normalization", "starting_stack"):
            value = self.config.get(key)
            if isinstance(value, (int, float)):
                preferred_scale = float(value)
                break
        return infer_normalization_scale(game, preferred_scale)

    @torch.no_grad()
    def choose_action(self, game: TexasHoldem, player_index: int):
        """Chooses an action by querying the model."""
        if self.model is None and self._fallback_strategy is not None:
            result = self._fallback_strategy.choose_action(game, player_index)
            if isinstance(result, tuple):
                action_str, amount = result
            else:
                action_str, amount = result, None
        else:
 
            normalization_scale = self._normalization_scale(game)

            prepare_fn = prepare_transformer_input
            supports_normalization = False
            try:
                signature = inspect.signature(prepare_fn)
            except (TypeError, ValueError):  # pragma: no cover - dynamic objects
                supports_normalization = False
            else:
                supports_normalization = "normalization_scale" in signature.parameters

            if supports_normalization:
                tensors = prepare_fn(
                    game,
                    player_index,
                    self.max_seq_len,
                    self.feature_dim,
                    normalization_scale=normalization_scale,
                )
            else:
                tensors = prepare_fn(
                    game,
                    player_index,
                    self.max_seq_len,
                    self.feature_dim,
                )

            hole = tensors[0]
            community = tensors[1]
            history = tensors[2]
            advantages = (
                self.model(
                    hole.unsqueeze(0).to(self._torch_device),
                    community.unsqueeze(0).to(self._torch_device),
                    history.unsqueeze(0).to(self._torch_device),
                )
                .squeeze(0)
                .cpu()
            )

            legal_mask = get_legal_actions_mask(game, player_index, self.num_actions)
            advantages[~legal_mask] = -1e9

            policy = calculate_strategy(advantages, self.num_actions)

            if policy.sum() > 0:
                action_idx = torch.multinomial(policy, 1).item()
            else:
                valid_indices = torch.where(legal_mask)[0]
                action_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()

            action = get_action_from_index(action_idx, game, player_id=player_index)
            action_str, amount = action_to_tuple(action)

        print(
            f"AI (Player {player_index + 1}) chose action: "
            f"{action_str} {amount if amount is not None else ''}"
        )
        return action_str, amount


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Play against a trained Poker AI model.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--starting-stack", type=int, default=1000, help="Starting stack size for players."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "npu"],
        default=None,
        help="Computation device. Defaults to CUDA if available, then NPU, then CPU.",
    )
    parser.add_argument(
        "--npu",
        action="store_true",
        help="Force NPU usage if available.",
    )
    parser.add_argument(
        "--use-all-npus",
        action="store_true",
        help="Wrap the model with DataParallel to utilize all NPUs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found at {args.model_path}")
        return

    use_all_npus = False

    # Set device
    if args.device:
        device = args.device
    elif args.npu:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = "npu"
        else:
            print("Warning: --npu specified but no NPUs available. Falling back to CPU.")
            device = "cpu"
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():
            device = "npu"
        else:
            device = "cpu"

    if (
        device == "npu"
        and args.use_all_npus
        and hasattr(torch, "npu")
        and torch.npu.is_available()
        and torch.npu.device_count() > 1
    ):
        use_all_npus = True
        print(f"Using all {torch.npu.device_count()} NPUs for inference.")

    print(f"Using device: {device}")

    # Instantiate strategies
    human_strategy = HumanStrategy()
    ai_strategy = AIStrategy(
        model_path=args.model_path,
        device=device,
        use_all_npus=use_all_npus,
    )

    # Set up the game
    # The game is for 2 players: one human, one AI
    game = TexasHoldem(
        num_players=2,
        starting_stack=args.starting_stack,
        player_strategies=[human_strategy, ai_strategy],
    )

    print("--- Welcome to Human vs AI Poker ---")
    print("You are Player 1. The AI is Player 2.")
    print(f"Model used: {args.model_path}")

    # Game loop
    try:
        while True:
            game.initialize_game()
            game.play_hand()
            print("\n--- Hand Over ---")
            # Ask user if they want to play another hand
            play_again = input("Play another hand? (y/n): ").lower()
            if play_again != "y":
                break
    except KeyboardInterrupt:
        print("\nGame terminated. Thanks for playing!")


if __name__ == "__main__":
    main()
