# human_vs_ai.py

"""Command line game allowing humans to play against simple AI players."""

import argparse
import sys
from typing import Any, cast

from poker_ai.ai import load_model_strategy
from poker_ai.config import load_config
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import (
    HumanStrategy,
    ModelAIStrategy,
    RandomAIStrategy,
)


def parse_args() -> argparse.Namespace:
    """Return parsed command line arguments."""
    parser = argparse.ArgumentParser(description="Play Texas Hold'em against simple AI players")
    parser.add_argument("--total-players", type=int, help="Total number of players (2-10)")
    parser.add_argument("--num-humans", type=int, help="Number of human players")
    parser.add_argument("--starting-stack", type=int, help="Starting chip count")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved AdvantageNetwork weights (.pth) to control AI players",
    )
 
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="Number of hands to play before exiting (0 for infinite play)",
    )
 
    parser.add_argument("--config", default=None, help="Path to configuration YAML file")
 
    return parser.parse_args()


def load_model(model_path: str) -> ModelAIStrategy | None:
    """Load a saved AI model using :func:`load_model_strategy`."""

    strategy, _ = load_model_strategy(model_path)
    if isinstance(strategy, ModelAIStrategy):
        return strategy
    return None


def main() -> None:  # noqa: C901
    args = parse_args()
    print("=== Welcome to Texas Hold'em Poker Simulation ===\n")

    cfg = load_config(args.config)

    model_strategy = load_model(args.model_path) if args.model_path else None

    total_players = args.total_players
    if total_players is None:
        while True:
            try:
                total_players = int(input("Enter the total number of players (2 to 10): "))
                if 2 <= total_players <= 10:
                    break
                else:
                    print("Total number of players must be between 2 and 10. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 2 and 10.")
    else:
        if not 2 <= total_players <= 10:
            raise ValueError("--total-players must be between 2 and 10")

    num_humans = args.num_humans
    if num_humans is None:
        while True:
            try:
                num_humans = int(
                    input(f"Enter the number of human players (0 to {total_players}): ")
                )
                if 0 <= num_humans <= total_players:
                    break
                else:
                    print(
                        "Number of human players must be between 0 and "
                        f"{total_players}. Please try again."
                    )
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    else:
        if not 0 <= num_humans <= total_players:
            raise ValueError("--num-humans must be between 0 and total players")

    num_ai = total_players - num_humans

    # Choose tournament type and set starting stack
    starting_stack = (
        args.starting_stack
        if args.starting_stack is not None
        else cfg.get("game_engine", {}).get("starting_stack", 10000)
    )
    if starting_stack is None:
        print("\nChoose tournament type:")
        print("1. Standard Tournament (10,000 chips)")
        choice = input("Enter choice (1): ")
        if choice == "1":
            starting_stack = 10000
        else:
            print("Invalid choice. Defaulting to Standard Tournament.")
            starting_stack = 10000

    # Create player strategies
    player_strategies: list[Any] = []
    for _ in range(num_humans):
        player_strategies.append(HumanStrategy())
    for _ in range(num_ai):
        if model_strategy:
            player_strategies.append(model_strategy)
        else:
            player_strategies.append(RandomAIStrategy())

    # Instantiate the game with chosen starting stack
    game = TexasHoldem(total_players, starting_stack, cast(list[Any], player_strategies))

    hands_to_play = args.num_hands
    hands_played = 0

    try:
        while hands_to_play == 0 or hands_played < hands_to_play:
            game.play_game()
            winner_info = getattr(game, "last_winner", None)
            if isinstance(winner_info, list) and winner_info:
                players = ", ".join(f"Player {idx + 1}" for idx in winner_info)
                print(f"Hand Summary: Tie between {players}.")
            elif isinstance(winner_info, int):
                print(f"Hand Summary: Player {winner_info + 1} wins!")
            else:
                print("Hand Summary: No winner determined.")
            print("\n--- Hand Completed ---")
            hands_played += 1
            if hands_to_play != 0 and hands_played >= hands_to_play:
                break
            print("Resetting chips and starting a new hand.\n")
            game.reset_for_next_hand()
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
        sys.exit()


if __name__ == "__main__":
    main()
