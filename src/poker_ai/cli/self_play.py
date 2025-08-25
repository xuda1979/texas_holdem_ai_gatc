import os
import sys
import time
import json
import random
import threading
import signal
import torch
from datetime import datetime
from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.config import config
from poker_ai.utils.state_representation import prepare_transformer_input
from poker_ai.utils.action_mapping import (
    get_action_from_index,
    get_legal_actions_mask,
)


class TransformerStrategy:
    """Wraps an ``AdvantageNetwork`` with game logic helpers."""

    def __init__(self, model: AdvantageNetwork, config: dict, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

    @property
    def is_human(self) -> bool:
        return False

    @torch.no_grad()
    def choose_action(self, game: TexasHoldem, player_index: int):
        max_seq_len = self.config.get("max_seq_len", 256)
        d_raw_feature = self.config.get(
            "d_raw_feature", self.config.get("input_feature_dim", 18)
        )
        state_tensor = prepare_transformer_input(
            game, player_index, max_seq_len, d_raw_feature
        )
        advantages = (
            self.model(state_tensor.unsqueeze(0).to(self.device))
            .squeeze(0)
            .cpu()
        )
        num_actions = self.config.get("num_actions", self.model.num_actions)
        legal_mask = get_legal_actions_mask(game, player_index, num_actions)

        # Mask out illegal actions and perform regret matching manually so that
        # the uniform fallback covers only legal moves.
        advantages[~legal_mask] = -float("inf")
        positive = torch.clamp(advantages, min=0) * legal_mask.float()
        if positive.sum() > 0:
            policy = positive / positive.sum()
        else:
            policy = legal_mask.float() / legal_mask.sum()

        action_idx = torch.multinomial(policy, 1).item()
        return get_action_from_index(action_idx, game, player_index)

COMMON_ACTIONS = ['talk', 'move']

def load_transformer_model():
    model_name = "texas_holdem_transformer_ai"
    weights_path, config_path = get_model_paths(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_exists(weights_path, config_path):
        return load_existing_model(weights_path, config_path, device)
    return initialize_new_model(device)

def get_model_paths(model_name):
    weights_path = os.path.join(config.MODEL_DIR, f"{model_name}.pth")
    config_path = os.path.join(config.MODEL_DIR, f'{model_name}_config.json')
    return weights_path, config_path

def model_exists(weights_path, config_path):
    return os.path.exists(weights_path) and os.path.exists(config_path)

def load_existing_model(weights_path, config_path, device):
    with open(config_path, "r") as f:
        model_config = json.load(f)
    network_params = {
        k: model_config[k]
        for k in [
            "input_feature_dim",
            "hidden_dim",
            "num_heads",
            "num_layers",
            "num_actions",
        ]
    }
    model = AdvantageNetwork(**network_params)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Model loaded from {weights_path}")
    return TransformerStrategy(model, model_config, device)

def initialize_new_model(device):
    print("No existing model found. Initializing a new model.")
    model_config = {
        "input_feature_dim": 18,
        "hidden_dim": 128,
        "num_heads": 2,
        "num_layers": 2,
        "num_actions": 10,
        "max_seq_len": 256,
        "d_raw_feature": 18,
    }
    network_params = {
        "input_feature_dim": model_config["input_feature_dim"],
        "hidden_dim": model_config["hidden_dim"],
        "num_heads": model_config["num_heads"],
        "num_layers": model_config["num_layers"],
        "num_actions": model_config["num_actions"],
    }
    model = AdvantageNetwork(**network_params)
    return TransformerStrategy(model, model_config, device)

def save_transformer_model(transformer_strategy):
    models_dir = config.MODEL_DIR
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    weight_path, config_path = get_save_paths()

    save_weights(transformer_strategy, weight_path)
    save_config(transformer_strategy, config_path)

def get_save_paths():
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    weight_path = os.path.join(config.MODEL_DIR, f"{timestamp}_model.pth")
    config_path = os.path.join(config.MODEL_DIR, f"{timestamp}_model.config.json")
    return weight_path, config_path

def save_weights(transformer_strategy, weight_path):
    try:
        torch.save(transformer_strategy.model.state_dict(), weight_path)
        print(f"Saved model weights to {weight_path}")
    except Exception as e:
        print(f"Error saving model weights: {e}")

def save_config(transformer_strategy, config_path):
    try:
        with open(config_path, "w") as f:
            json.dump(transformer_strategy.config, f, indent=4)
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
