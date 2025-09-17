"""
A command-line interface to play against a trained Poker AI model.
"""

import argparse
import os

import torch

# ruff: noqa: ANN201,ANN204
from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import HumanStrategy, PlayerStrategy
from poker_ai.rules.cfr import calculate_strategy
from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import prepare_transformer_input


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

        # This assumes the model was saved with a config that matches the network class
        # For now, we hardcode the model parameters, but a config file would be better.
        self.model = AdvantageNetwork(
            history_feature_dim=18,  # This must match the state representation
            card_feature_dim=18,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            num_actions=self.num_actions,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        if (
            use_all_npus
            and device == "npu"
            and hasattr(torch, "npu")
            and torch.npu.is_available()
            and torch.npu.device_count() > 1
        ):
            # Wrap model for multi-NPU inference
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()

    @property
    def is_human(self):
        return False

    @torch.no_grad()
    def choose_action(self, game: TexasHoldem, player_index: int):
        """Chooses an action by querying the model."""
        # 1. Get the policy from the network
        hole, community, history = prepare_transformer_input(game, player_index, 256, 18)
        advantages = (
            self.model(
                hole.unsqueeze(0).to(self.device),
                community.unsqueeze(0).to(self.device),
                history.unsqueeze(0).to(self.device),
            )
            .squeeze(0)
            .cpu()
        )

        # 2. Get legal actions and mask the policy
        legal_mask = get_legal_actions_mask(game, player_index, self.num_actions)
        advantages[~legal_mask] = -1e9

        policy = calculate_strategy(advantages, self.num_actions)

        # 3. Sample an action from the policy
        if policy.sum() > 0:
            action_idx = torch.multinomial(policy, 1).item()
        else:
            # Fallback if policy is all zeros (should not happen with legal mask)
            valid_indices = torch.where(legal_mask)[0]
            action_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()

        # 4. Convert action index to game action
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
