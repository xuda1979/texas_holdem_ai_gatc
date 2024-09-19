# import os
# import sys
# import time
# import json
# import random
# import threading
# import signal
# from datetime import datetime

# import tensorflow as tf
# from models.transformer_strategy_model import TransformerAIStrategy, TransformerStrategyModel
# from game_engine.texas_holdem import TexasHoldem
# import config

# def load_transformer_model():
#     model_name = 'texas_holdem_transformer_ai'
#     num_actions = 6  # Define the number of actions (e.g., fold, check, call, bet, raise, all-in)
    
#     # Check if weights and config files exist
#     weights_path = os.path.join(config.MODEL_DIR, f'{model_name}.weights.h5')
#     config_path = os.path.join(config.MODEL_DIR, f'{model_name}_config.json')

#     if os.path.exists(weights_path) and os.path.exists(config_path):
#         # Load model configuration
#         with open(config_path, 'r') as f:
#             model_config = json.load(f)
        
#         # Rebuild the model
#         model = TransformerStrategyModel.from_config(model_config)
#         model.build(input_shape=(None, 1, model_config['d_model']))  # Adjust input shape as needed
#         model.load_weights(weights_path)
#         print(f'Model loaded from {weights_path}')
#         return TransformerAIStrategy(num_actions=num_actions, model=model)
#     else:
#         print('No existing model found. Initializing a new model.')
#         return TransformerAIStrategy(num_actions=num_actions)

# def save_transformer_model(transformer_strategy):
#     models_dir = config.MODEL_DIR
#     if not os.path.exists(models_dir):
#         os.makedirs(models_dir)
#         print(f"Created directory for models at {models_dir}")

#     timestamp = datetime.now().strftime("%y%m%d%H%M%S")
#     weight_filename = f"{timestamp}_model.weights.h5"
#     config_filename = f"{timestamp}_model.config.json"

#     weight_path = os.path.join(models_dir, weight_filename)
#     config_path = os.path.join(models_dir, config_filename)

#     # Save weights
#     try:
#         transformer_strategy.model.save_weights(weight_path)
#         print(f"Saved model weights to {weight_path}")
#     except Exception as e:
#         print(f"Error saving model weights: {e}")

#     # Save config
#     try:
#         config = transformer_strategy.model.get_config()
#         with open(config_path, 'w') as f:
#             json.dump(config, f, indent=4)
#         print(f"Saved model configuration to {config_path}")
#     except Exception as e:
#         print(f"Error saving model configuration: {e}")

# def save_game_history(game):
#     history_dir = os.path.join(config.BASE_DATA_DIR, 'historical_actions')
#     if not os.path.exists(history_dir):
#         os.makedirs(history_dir)
#         print(f"Created directory for game histories at {history_dir}")

#     timestamp = datetime.now().strftime("%y%m%d%H%M%S")
#     filename = f"{timestamp}_simulated_game.json"
#     filepath = os.path.join(history_dir, filename)

#     # Prepare game data
#     game_data = {
#         'hand_number': game.hand_count,
#         'dealer': game.rules.dealer_button + 1,
#         'actions': game.rules.betting_history,
#         'community_cards': game.rules.community_cards,
#         'pot': game.rules.pot,
#         'players': game.get_player_status()
#     }

#     # Save the game data as JSON
#     try:
#         with open(filepath, 'w') as f:
#             json.dump(game_data, f, indent=4)
#         print(f"Saved simulated game history to {filepath}")
#     except Exception as e:
#         print(f"Error saving game history: {e}")

# def simulate_game(transformer_strategy):
#     num_players = random.randint(2, 10)
#     tournament_type = 'Standard Tournament'
#     starting_stack = 10000  # Default for Standard Tournament

#     print(f"\n--- Starting {tournament_type} with {num_players} AI players ---")

#     player_strategies = [transformer_strategy] * num_players
#     game = TexasHoldem(num_players, starting_stack, player_strategies)

#     game.play_game()
#     save_game_history(game)

# def periodic_save(transformer_strategy, interval=1800):
#     def save_loop():
#         while True:
#             time.sleep(interval)
#             print("\n[Periodic Save] Saving Transformer model...")
#             save_transformer_model(transformer_strategy)
#             print("[Periodic Save] Model saved successfully.\n")

#     save_thread = threading.Thread(target=save_loop, daemon=True)
#     save_thread.start()
#     print(f"Started periodic model saving every {interval / 60} minutes.")

# def handle_termination(transformer_strategy):
#     def signal_handler(sig, frame):
#         print("\n[Termination] Termination signal received. Saving model before exit...")
#         save_transformer_model(transformer_strategy)
#         print("[Termination] Model saved. Exiting now.")
#         sys.exit(0)

#     signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
#     signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
#     print("Signal handlers for termination set up.")

# def main():
#     # Load or initialize the Transformer model
#     transformer_strategy = load_transformer_model()

#     # Set up periodic model saving every 30 minutes (1800 seconds)
#     periodic_save(transformer_strategy, interval=1800)

#     # Set up signal handlers for graceful termination
#     handle_termination(transformer_strategy)

#     print("Starting self-play simulation. Press Ctrl+C to terminate.")

#     # Continuous self-play loop
#     while True:
#         simulate_game(transformer_strategy)
#         # Optional: Add a short delay between games to manage resource usage
#         time.sleep(1)  # Sleep for 1 second

# if __name__ == "__main__":
#     main()


import os
import sys
import time
import json
import random
import threading
import signal
from datetime import datetime
import tensorflow as tf
from models.transformer_strategy_model import TransformerAIStrategy, TransformerStrategyModel
from game_engine.texas_holdem import TexasHoldem
import config

COMMON_ACTIONS = ['talk', 'move']

def load_transformer_model():
    model_name = 'texas_holdem_transformer_ai'
    weights_path, config_path = get_model_paths(model_name)
    
    if model_exists(weights_path, config_path):
        return load_existing_model(weights_path, config_path)
    return initialize_new_model()

def get_model_paths(model_name):
    weights_path = os.path.join(config.MODEL_DIR, f'{model_name}.weights.h5')
    config_path = os.path.join(config.MODEL_DIR, f'{model_name}_config.json')
    return weights_path, config_path

def model_exists(weights_path, config_path):
    return os.path.exists(weights_path) and os.path.exists(config_path)

def load_existing_model(weights_path, config_path):
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    model = TransformerStrategyModel.from_config(model_config)
    model.build(input_shape=(None, 1, model_config['d_model']))
    model.load_weights(weights_path)
    print(f'Model loaded from {weights_path}')
    return TransformerAIStrategy(model=model)

def initialize_new_model():
    print('No existing model found. Initializing a new model.')
    return TransformerAIStrategy()

def save_transformer_model(transformer_strategy):
    models_dir = config.MODEL_DIR
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    weight_path, config_path = get_save_paths()

    save_weights(transformer_strategy, weight_path)
    save_config(transformer_strategy, config_path)

def get_save_paths():
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    weight_path = os.path.join(config.MODEL_DIR, f"{timestamp}_model.weights.h5")
    config_path = os.path.join(config.MODEL_DIR, f"{timestamp}_model.config.json")
    return weight_path, config_path

def save_weights(transformer_strategy, weight_path):
    try:
        transformer_strategy.model.save_weights(weight_path)
        print(f"Saved model weights to {weight_path}")
    except Exception as e:
        print(f"Error saving model weights: {e}")

def save_config(transformer_strategy, config_path):
    try:
        config = transformer_strategy.model.get_config()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved model configuration to {config_path}")
    except Exception as e:
        print(f"Error saving model configuration: {e}")

def append_common_actions(actions):
    return actions + COMMON_ACTIONS

def save_game_history(game):
    history_dir = os.path.join(config.BASE_DATA_DIR, 'historical_actions')
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    
    filepath = get_history_filepath()
    save_game_to_file(game, filepath)

def get_history_filepath():
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    return os.path.join(config.BASE_DATA_DIR, 'historical_actions', f"{timestamp}_simulated_game.json")

def save_game_to_file(game, filepath):
    game_data = extract_game_data(game)
    try:
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=4)
        print(f"Saved simulated game history to {filepath}")
    except Exception as e:
        print(f"Error saving game history: {e}")

def extract_game_data(game):
    return {
        'hand_number': game.hand_count,
        'dealer': game.rules.dealer_button + 1,
        'actions': append_common_actions(game.rules.betting_history),
        'community_cards': game.rules.community_cards,
        'pot': game.rules.pot,
        'players': game.get_player_status()
    }

def simulate_game(transformer_strategy):
    num_players = random.randint(2, 10)
    starting_stack = 10000
    print(f"\n--- Starting game with {num_players} AI players ---")

    player_strategies = [transformer_strategy] * num_players
    game = TexasHoldem(num_players, starting_stack, player_strategies)
    game.play_game()
    save_game_history(game)

def periodic_save(transformer_strategy, interval=1800):
    save_thread = threading.Thread(target=save_loop, args=(transformer_strategy, interval), daemon=True)
    save_thread.start()
    print(f"Started periodic model saving every {interval / 60} minutes.")

def save_loop(transformer_strategy, interval):
    while True:
        time.sleep(interval)
        print("\n[Periodic Save] Saving Transformer model...")
        save_transformer_model(transformer_strategy)
        print("[Periodic Save] Model saved successfully.\n")

def handle_termination(transformer_strategy):
    signal.signal(signal.SIGINT, lambda sig, frame: terminate_gracefully(transformer_strategy))
    signal.signal(signal.SIGTERM, lambda sig, frame: terminate_gracefully(transformer_strategy))
    print("Signal handlers for termination set up.")

def terminate_gracefully(transformer_strategy):
    print("\n[Termination] Saving model before exit...")
    save_transformer_model(transformer_strategy)
    print("[Termination] Model saved. Exiting now.")
    sys.exit(0)

def main():
    transformer_strategy = load_transformer_model()
    periodic_save(transformer_strategy, interval=1800)
    handle_termination(transformer_strategy)

    print("Starting self-play simulation. Press Ctrl+C to terminate.")
    while True:
        simulate_game(transformer_strategy)
        time.sleep(1)

if __name__ == "__main__":
    main()
