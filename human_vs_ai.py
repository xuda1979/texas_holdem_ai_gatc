# human_vs_ai.py

from game_engine.texas_holdem import TexasHoldem
from playStrategy import HumanStrategy, RandomAIStrategy
import sys

def main():
    print("=== Welcome to Texas Hold'em Poker Simulation ===\n")

    # Validate total number of players
    while True:
        try:
            total_players = int(input("Enter the total number of players (2 to 10): "))
            if 2 <= total_players <= 10:
                break
            else:
                print("Total number of players must be between 2 and 10. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a numeric value between 2 and 10.")

    # Validate number of human players
    while True:
        try:
            num_humans = int(input(f"Enter the number of human players (0 to {total_players}): "))
            if 0 <= num_humans <= total_players:
                break
            else:
                print(f"Number of human players must be between 0 and {total_players}. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    num_ai = total_players - num_humans

    # Choose tournament type and set starting stack
    print("\nChoose tournament type:")
    print("1. Standard Tournament (10,000 chips)")
    # Only one type since no rebuy
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
