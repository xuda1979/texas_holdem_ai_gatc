import os
import json
import tensorflow as tf
from config import MODEL_DIR, SIMULATED_DATA_DIR

from game_engine.texas_holdem import TexasHoldem
from models.transformer_strategy_model import TransformerStrategyModel
from cfr_algorithm.cfr_trainer import CFRTrainer
from self_play.self_play import SelfPlay

def save_model(model, model_name):
    # Normalize file paths for Windows compatibility
    weights_path = os.path.normpath(os.path.join(MODEL_DIR, f'{model_name}.weights.h5'))
    config_path = os.path.normpath(os.path.join(MODEL_DIR, f'{model_name}_config.json'))

    # Save model weights
    model.save_weights(weights_path)
    
    # Save model configuration
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f)
    print(f'Model saved: {weights_path} and {config_path}')


def load_model_if_exists(model_name, num_actions):
    weights_path = os.path.join(MODEL_DIR, f'{model_name}_weights.h5')
    config_path = os.path.join(MODEL_DIR, f'{model_name}_config.json')

    if os.path.exists(weights_path) and os.path.exists(config_path):
        # Load model configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Rebuild the model
        model = TransformerStrategyModel.from_config(config)
        model.build(input_shape=(None, 100, config['d_model']))  # Adjust input shape as needed
        model.load_weights(weights_path)
        print(f'Model loaded from {weights_path}')
        return model
    else:
        print('No existing model found. Initializing a new model.')
        return None

def train(num_games=1000, num_players=2, starting_stack=1000, num_actions=4):
    # Initialize game engine
    game = TexasHoldem(num_players=num_players, starting_stack=starting_stack)

    # Check if the model already exists
    model = load_model_if_exists('texas_holdem_cfr_transformer', num_actions)
    
    # If no model is found, create a new one
    if model is None:
        model = TransformerStrategyModel(num_actions=num_actions)
        save_model(model, 'texas_holdem_cfr_transformer')  # Save the new model
    
    # Initialize the CFR trainer with the model
    cfr_trainer = CFRTrainer(model, num_actions=num_actions)

    # Self-play for training
    self_play = SelfPlay(cfr_trainer, num_games=num_games)
    simulation_results = self_play.simulate_game(game)

    # Save the simulated results
    simulation_file_path = os.path.join(SIMULATED_DATA_DIR, 'simulation_results.json')
    with open(simulation_file_path, 'w') as f:
        json.dump(simulation_results, f)
    print(f"Simulated results saved to {simulation_file_path}")

if __name__ == "__main__":
    train()