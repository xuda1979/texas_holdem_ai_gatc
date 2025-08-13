"""Main command-line interface for training models via self-play."""

import yaml
import os # For path manipulation if needed, e.g. for robust config loading
import argparse
import random
import time
import torch

# Assuming the script is run from the project root,
# and trainers, self_play, etc., are packages in that root.
from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer
from poker_ai.selfplay.self_play import SelfPlay, load_transformer_model   # ← new
from typing import List, Dict

# Configuration Loading
# Robustly locate config.yaml assuming it's in the project root
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
# If script is not in root, adjust path:
# CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml") # If config is with script
# Or an absolute path, or environment variable. For now, assume it's in CWD.

def load_configuration(config_path: str) -> dict:
    """Loads YAML configuration from the given path."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            print(f"Warning: {config_path} is empty or invalid. Using default configurations.")
            return {} # Return empty dict to trigger defaults everywhere
        return config_data
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default configurations.")
        return {} # Return empty dict
    except yaml.YAMLError as e:
        print(f"Error parsing {config_path}: {e}. Using default configurations.")
        return {}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Run Poker AI training session")
    parser.add_argument(
        "--num-hands",
        type=int,
        help="Number of hands to simulate (overrides config)"
    )
    parser.add_argument(
        "--save-model-every",
        type=int,
        help="Save model every N hands"
    )
    parser.add_argument(
        "--save-minutes",
        type=int,
        help="Save model every N minutes (overrides config)"
    )
    parser.add_argument(
        "--algorithm",
        default="ai_cfr",
        choices=["ai_cfr", "deep_cfr", "single_network"],
        help="Training algorithm to use"
    )
    parser.add_argument(
        "--config",
        default=CONFIG_FILE_PATH,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "npu"],
        default=None,
        help="Computation device. Defaults to CUDA if available, then NPU, then CPU.",
    )
    return parser.parse_args()


def initialize_trainer(algorithm: str, config: dict, device: str):
    """Return a trainer instance based on selected algorithm."""
    if algorithm == "ai_cfr":
        return AICFRTrainer(device=device)
    elif algorithm == "deep_cfr":
        from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer
        model_cfg = config.get("model", {})
        d_raw = model_cfg.get("d_raw_feature", 3)
        hidden = model_cfg.get("hidden_dim", 128)
        num_actions = model_cfg.get("num_actions", 10)
        lr = model_cfg.get("learning_rate", 1e-3)
        return DeepCFRTrainer(d_raw, hidden, num_actions, learning_rate=lr, device=device)
    elif algorithm == "single_network":
        from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer
        model_cfg = config.get("model", {})
        d_raw = model_cfg.get("d_raw_feature", 3)
        hidden = model_cfg.get("hidden_dim", 128)
        num_actions = model_cfg.get("num_actions", 10)
        lr = model_cfg.get("learning_rate", 1e-3)
        return SingleNetworkCFRTrainer(d_raw, hidden, num_actions, lr, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    print("--- Starting Poker AI Training Session ---")

    args = parse_args()
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"
        else:
            device = "cpu"

    # Load configuration
    config = load_configuration(args.config)

    # Extract configurations with defaults
    # model_config is implicitly used by AICFRTrainer via its own global config load.
    # We don't directly use model_config here, but AICFRTrainer does.
    
    game_engine_config = config.get('game_engine', {})
    training_params = config.get('training', {})
    curriculum_stages: List[Dict] = config.get('curriculum', {}).get('stages', [])

    # Training Parameters with CLI overrides
    # In MCCFR, each "hand" is one full traversal, which is one iteration.
    num_iterations = (
        args.num_hands if args.num_hands is not None else training_params.get('num_training_hands', 10000)
    )
    save_model_every_n_hands = (
        args.save_model_every if args.save_model_every is not None else training_params.get('save_model_every_n_hands', 1000)
    )
    save_model_every_minutes = (
        args.save_minutes if args.save_minutes is not None
        else training_params.get('save_model_every_minutes', 10) # Default to 10 if not in config
    )
    
    # Game Engine Parameters for SelfPlay
    num_players = game_engine_config.get('num_players', 2)
    starting_stack = game_engine_config.get('starting_stack', 1000)
    big_blind = game_engine_config.get('big_blind', 10)
    small_blind = game_engine_config.get('small_blind', 5)

    print("\n--- Configuration ---")
    print(f"Total training iterations: {num_iterations}")
    print(f"Save model every: {save_model_every_n_hands} hands (if >0)")
    print(f"Save model every: {save_model_every_minutes} minutes (if >0)")
    print(f"Number of players: {num_players}")
    print(f"Starting stack: {starting_stack}")
    print(f"Blinds: SB={small_blind}, BB={big_blind}")
    print(f"Using device: {device}")
    # Note: The new DeepCFRTrainer and SelfPlay are simplified for 2-player HU NLHE.
    # We will enforce this here.
    if num_players != 2 and args.algorithm == 'deep_cfr':
        print("Warning: The refactored 'deep_cfr' algorithm is designed for 2 players.")
        print("Setting number of players to 2 for this training session.")
        num_players = 2

    # Initialization
    print("\n--- Initializing Components ---")
    game_config_for_selfplay = {
        'num_players': num_players,
        'starting_stack': starting_stack,
        'big_blind': big_blind,
        'small_blind': small_blind
    }

    try:
        cfr_trainer = initialize_trainer(args.algorithm, config, device)
        print(f"{args.algorithm} trainer initialized.")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # The new SelfPlay class for MCCFR doesn't need curriculum learning or complex setup.
    # It's simplified for the core algorithm.
    self_play_env = SelfPlay(cfr_trainer=cfr_trainer, game_engine_config=game_config_for_selfplay)

    # Training Loop
    print("\n--- Starting Training Loop ---")
    last_save_time = time.time()

    for iteration in range(1, num_iterations + 1):
        print(f"\n--- MCCFR Iteration {iteration}/{num_iterations} ---")
        try:
            # The play_hand_for_training method now runs one MCCFR traversal
            # and triggers the training step internally.
            self_play_env.play_hand_for_training(iteration)
        except Exception as e:
            print(f"Error during iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            # Decide if training should continue or break on error
            # For now, we break on error as it might indicate a deeper issue.
            break

        # Check conditions for saving model
        current_time = time.time()
        time_since_last_save_minutes = (current_time - last_save_time) / 60

        hand_save_condition_met = (save_model_every_n_hands > 0 and iteration % save_model_every_n_hands == 0)
        time_save_condition_met = (save_model_every_minutes > 0 and time_since_last_save_minutes >= save_model_every_minutes)

        if hand_save_condition_met or time_save_condition_met:
            print(f"\n--- Saving model at iteration {iteration} ---")
            if hand_save_condition_met:
                print(f"Reason: Iteration count ({save_model_every_n_hands} iterations interval reached)")
            if time_save_condition_met:
                print(f"Reason: Time interval ({save_model_every_minutes} minutes interval reached)")

            # The save_model method in the new trainer expects a path.
            # We'll create a simple path based on the algorithm and iteration.
            model_save_path = f"models/{args.algorithm}_iteration_{iteration}.pth"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            try:
                cfr_trainer.save_model(model_save_path)
                print(f"Model saved successfully to {model_save_path}")
                last_save_time = current_time
            except Exception as e:
                print(f"Error saving model at iteration {iteration}: {e}")
    
    # Final save after the loop
    print("\n--- Training session finished ---")
    print("Saving final model...")
    final_model_path = f"models/{args.algorithm}_final.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    try:
        cfr_trainer.save_model(final_model_path)
        print(f"Final model saved successfully to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    print("\n--- Training Complete ---")

if __name__ == '__main__':
    main()
