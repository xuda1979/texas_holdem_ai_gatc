"""
A command-line interface to play against a trained Poker AI model.
"""

import argparse
import os

import torch

# ruff: noqa: ANN201,ANN204
from poker_ai.ai.model_loader import load_model_strategy
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

        if isinstance(loaded_strategy, ModelAIStrategy):
            model = loaded_strategy.model.to(target_device)

            if (
                use_all_npus
                and target_device.type == "npu"
                and hasattr(torch, "npu")
                and torch.npu.is_available()
                and torch.npu.device_count() > 1
            ):
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
 
            hole, community, history = prepare_transformer_input(
                game,
                player_index,
                self.max_seq_len,
                self.feature_dim,
 
                normalization_scale=normalization_scale,
 
            )
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
