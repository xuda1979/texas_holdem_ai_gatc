# human_vs_ai.py

"""Command line game allowing humans to play against simple AI players."""

import argparse
import sys
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import HumanStrategy, RandomAIStrategy


def parse_args() -> argparse.Namespace:
    """Return parsed command line arguments."""
    parser = argparse.ArgumentParser(description="Play Texas Hold'em against simple AI players")
    parser.add_argument("--total-players", type=int, help="Total number of players (2-10)")
    parser.add_argument("--num-humans", type=int, help="Number of human players")
    parser.add_argument("--starting-stack", type=int, default=10000, help="Starting chip count")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print("=== Welcome to Texas Hold'em Poker Simulation ===\n")

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
                num_humans = int(input(f"Enter the number of human players (0 to {total_players}): "))
                if 0 <= num_humans <= total_players:
                    break
                else:
                    print(f"Number of human players must be between 0 and {total_players}. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    else:
        if not 0 <= num_humans <= total_players:
            raise ValueError("--num-humans must be between 0 and total players")

    num_ai = total_players - num_humans

    # Choose tournament type and set starting stack
    starting_stack = args.starting_stack
    if starting_stack is None:
        print("\nChoose tournament type:")
        print("1. Standard Tournament (10,000 chips)")
        choice = input("Enter choice (1): ")
        if choice == '1':
            starting_stack = 10000
        else:
            print("Invalid choice. Defaulting to Standard Tournament.")
            starting_stack = 10000

    # Create player strategies
    player_strategies = []
    for i in range(num_humans):
        player_strategies.append(HumanStrategy())
    for i in range(num_ai):
        player_strategies.append(RandomAIStrategy())

    # Instantiate the game with chosen starting stack
    game = TexasHoldem(total_players, starting_stack, player_strategies)

    # Play the game indefinitely
    try:
        while True:
            game.play_game()
            print("\n--- Hand Completed ---")
            print("Resetting chips and starting a new hand.\n")
            game.reset_for_next_hand()
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
        sys.exit()

if __name__ == "__main__":
    main()
