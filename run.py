import sys
import random
from texas_holdem import TexasHoldem
from cfr_trainer import CFRTrainer

def play_game(num_players, num_human_players, strategy):
    """Play a Texas Hold'em game with AI and human players."""
    game = TexasHoldem(num_players)
    game.reset()

    human_players = list(range(num_human_players))
    ai_players = list(range(num_human_players, num_players))

    for player_id in range(num_players):
        if player_id in human_players:
            action = input(f"Player {player_id+1} (Human), enter your action (fold/call): ").strip().lower()
        else:
            # AI agent selects action based on the provided strategy
            if random.random() < strategy.get('call', 0.5):
                action = 'call'
            else:
                action = 'fold'
            print(f"Player {player_id+1} (AI) chooses to {action}")

        game.apply_action(player_id, action)

    winners = game.get_winner()
    print(f"Winner(s): {', '.join([f'Player {w+1}' for w in winners])}")

def train_model(num_players, iterations, model_path="cfr_model.pkl"):
    """Train the AI agent using CFR."""
    trainer = CFRTrainer(num_players, iterations, model_path)
    trainer.train()
    optimal_strategy = trainer.get_optimal_strategy()
    print("Optimal strategy learned:")
    for info_set, strategy in optimal_strategy.items():
        print(f"{info_set}: {strategy}")
    return optimal_strategy

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|play]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        num_players = int(input("Enter the number of players: "))
        iterations = int(input("Enter the number of iterations for training: "))
        train_model(num_players, iterations)
    elif command == "play":
        num_players = int(input("Enter the total number of players: "))
        num_human_players = int(input("Enter the number of human players: "))
        if num_human_players > num_players:
            print("Number of human players cannot exceed total number of players.")
            sys.exit(1)
        
        # Load the trained strategy from the model file
        model_path = "cfr_model.pkl"
        trainer = CFRTrainer(num_players, 0, model_path)
        trainer.load_model()
        strategy = trainer.get_optimal_strategy()

        play_game(num_players, num_human_players, strategy)
    else:
        print("Unknown command. Use 'train' or 'play'.")
