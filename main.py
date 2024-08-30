import argparse
from cfr_trainer import CFRTrainer
from texas_holdem import TexasHoldem

def train_model(config, iterations, model_save_path):
    trainer = CFRTrainer(config)
    trainer.train(iterations=iterations)
    trainer.save_model(model_save_path)
    print(f"Model trained and saved to {model_save_path}")

def play_game(config, model_path, num_players=2):
    trainer = CFRTrainer(config)
    trainer.load_model(model_path)
    
    game = TexasHoldem(num_players=num_players)
    game.play_round()
    
    winner, best_hand = game.determine_winner()
    print(f"The winner is Player {winner} with the hand: {best_hand}")

def main():
    parser = argparse.ArgumentParser(description="Train or play Texas Hold'em using CFR and deep learning.")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--play', action='store_true', help="Play a game with the trained model")
    parser.add_argument('--iterations', type=int, default=1000, help="Number of training iterations")
    parser.add_argument('--model', type=str, default='texas_holdem_model.h5', help="Path to save/load the model")
    parser.add_argument('--num_players', type=int, default=2, help="Number of players in the game")

    args = parser.parse_args()

    config = {
        'input_shape': (10, 10, 1),  # Update according to your game state encoding
        'num_actions': 10,
        'learning_rate': 0.001,
        'num_res_blocks': 3,
        'num_players': args.num_players
    }

    if args.train:
        train_model(config, args.iterations, args.model)
    elif args.play:
        play_game(config, args.model, num_players=args.num_players)
    else:
        print("Please specify either --train or --play")

if __name__ == "__main__":
    main()
