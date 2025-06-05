import yaml
import os # For path manipulation if needed, e.g. for robust config loading
import argparse # Added for command-line arguments
import sys # Added for debug exit
from trainers.ai_cfr_trainer import AICFRTrainer
from self_play.self_play import SelfPlay
from typing import List, Dict

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


def main():
    print("--- Starting Poker AI Training Session ---")

    parser = argparse.ArgumentParser(description="Run Poker AI Training Session")
    parser.add_argument('--config-file', type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument('--num-training-hands', type=int, help="Total number of hands to train.")
    parser.add_argument('--save-model-every', type=int, help="Save the model every N hands.")
    parser.add_argument('--learning-rate', type=float, help="Learning rate for the optimizer.")
    parser.add_argument('--hidden-dim', type=int, help="Hidden dimension for the model.")
    parser.add_argument('--num-layers', type=int, help="Number of layers in the Transformer model.")
    parser.add_argument('--num-actions', type=int, help="Number of possible actions.")
    parser.add_argument('--d-raw-feature', type=int, help="Dimension of raw features for state representation.")
    args = parser.parse_args()

    print(f"DEBUG: Parsed arguments: {args}") # Added debug print
    # if args.num_training_hands == 12345:
    #     print("DEBUG: Test for --num-training-hands 12345 successful. Exiting.")
    #     sys.exit(0)
    # if args.learning_rate == 0.12345:
    #     print("DEBUG: Test for --learning-rate 0.12345 successful. Exiting.")
    #     sys.exit(0)

    # Load base configuration from YAML
    config = load_configuration(args.config_file)

    # Ensure 'training' and 'model' keys exist, defaulting to empty dicts if not
    if 'training' not in config:
        config['training'] = {}
    if 'model' not in config:
        config['model'] = {}
    if 'game_engine' not in config:
        config['game_engine'] = {}
    if 'curriculum' not in config:
        config['curriculum'] = {}
    if 'stages' not in config['curriculum']:
        config['curriculum']['stages'] = []


    # Override with command-line arguments if provided
    if args.num_training_hands is not None:
        config['training']['num_training_hands'] = args.num_training_hands
    if args.save_model_every is not None:
        config['training']['save_model_every_n_hands'] = args.save_model_every
    if args.learning_rate is not None:
        config['model']['learning_rate'] = args.learning_rate
    if args.hidden_dim is not None:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.num_layers is not None:
        config['model']['num_layers'] = args.num_layers
    if args.num_actions is not None:
        config['model']['num_actions'] = args.num_actions
    if args.d_raw_feature is not None:
        config['model']['d_raw_feature'] = args.d_raw_feature

    print("DEBUG: --- Final Configuration Values ---")
    print(f"DEBUG: Using config file: {args.config_file}")
    print(f"DEBUG: Final num_training_hands: {config.get('training', {}).get('num_training_hands')}")
    print(f"DEBUG: Final save_model_every_n_hands: {config.get('training', {}).get('save_model_every_n_hands')}")
    print(f"DEBUG: Final learning_rate: {config.get('model', {}).get('learning_rate')}")
    print(f"DEBUG: Final hidden_dim: {config.get('model', {}).get('hidden_dim')}")
    print(f"DEBUG: Final num_layers: {config.get('model', {}).get('num_layers')}")
    print(f"DEBUG: Final num_actions: {config.get('model', {}).get('num_actions')}")
    print(f"DEBUG: Final d_raw_feature: {config.get('model', {}).get('d_raw_feature')}")
    print(f"DEBUG: Final save_model_path: {config.get('training', {}).get('save_model_path')}")
    print("DEBUG: --- End of Final Configuration Values ---")

    # Extract configurations with defaults from the consolidated config
    training_params = config.get('training', {})
    game_engine_config_params = config.get('game_engine', {})
    curriculum_stages: List[Dict] = config.get('curriculum', {}).get('stages', [])

    # Training Parameters
    num_training_hands = training_params.get('num_training_hands', 1000)
    save_model_every_n_hands = training_params.get('save_model_every_n_hands', 100)
    
    # Game Engine Parameters for SelfPlay
    num_players = game_engine_config_params.get('num_players', 2)
    starting_stack = game_engine_config_params.get('starting_stack', 1000)
    big_blind = game_engine_config_params.get('big_blind', 10)
    small_blind = game_engine_config_params.get('small_blind', 5)

    print("\n--- Configuration ---")
    print(f"Config file used: {args.config_file}")
    print(f"Total training hands: {num_training_hands}")
    print(f"Save model every: {save_model_every_n_hands} hands")
    print(f"Number of players: {num_players}")
    print(f"Starting stack: {starting_stack}")
    print(f"Blinds: SB={small_blind}, BB={big_blind}")
    # Model parameters will be passed to AICFRTrainer
    print(f"Model Params (from config): Hidden Dim: {config.get('model', {}).get('hidden_dim')}, LR: {config.get('model', {}).get('learning_rate')}, etc.")


    # Initialization
    print("\n--- Initializing Components ---")
    # game_config_for_selfplay is now directly derived from the main config's game_engine section
    game_config_for_selfplay = {
        'num_players': num_players, # Already extracted, but can also use game_engine_config_params.get(...)
        'starting_stack': starting_stack,
        'big_blind': big_blind,
        'small_blind': small_blind,
        # Include any other game engine specific params from config['game_engine']
        **{k: v for k, v in game_engine_config_params.items() if k not in ['num_players', 'starting_stack', 'big_blind', 'small_blind']}
    }

    try:
        # Pass the consolidated config to AICFRTrainer
        cfr_trainer = AICFRTrainer(config_data=config)
        print("AICFRTrainer initialized.")
    except Exception as e:
        print(f"Error initializing AICFRTrainer: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure configuration (file and CLI args) is correct, especially for the 'model' section.")
        return

    try:
        self_play_env = SelfPlay(cfr_trainer=cfr_trainer, game_engine_config=game_config_for_selfplay)
        print("SelfPlay environment initialized.")
    except Exception as e:
        print(f"Error initializing SelfPlay environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Training Loop
    print("\n--- Starting Training Loop ---")
    print(f"DEBUG: Starting training loop with num_training_hands = {num_training_hands}") # Added debug print
    current_stage_index = 0
    if curriculum_stages:
        print(f"Applying initial curriculum stage: {curriculum_stages[current_stage_index]}")
        current_stage_config = curriculum_stages[current_stage_index]
        # Update game_config_for_selfplay with stage specific settings
        # Create a new dict for the stage to avoid modifying the base game_config_for_selfplay if not intended
        stage_game_config = game_config_for_selfplay.copy()
        stage_game_config.update(current_stage_config)
        self_play_env = SelfPlay(cfr_trainer=cfr_trainer, game_engine_config=stage_game_config)
        print(f"SelfPlay environment re-initialized with curriculum stage: {current_stage_config}")


    for hand_num in range(1, num_training_hands + 1):
        print(f"\n--- Training Hand {hand_num}/{num_training_hands} ---")

        # Curriculum update logic
        if curriculum_stages:
            # Calculate which stage this hand_num falls into
            # This is a simple way: divide total hands by number of stages to get interval
            # More sophisticated logic could be hand_num > stage_config.get('end_hand_num')
            hands_per_stage = num_training_hands // len(curriculum_stages)
            if hands_per_stage == 0: hands_per_stage = 1 # Avoid division by zero if num_training_hands < len(curriculum_stages)

            new_stage_index = (hand_num -1) // hands_per_stage # current stage index

            if new_stage_index != current_stage_index and new_stage_index < len(curriculum_stages):
                current_stage_index = new_stage_index
                print(f"Transitioning to curriculum stage {current_stage_index}: {curriculum_stages[current_stage_index]}")
                current_stage_config = curriculum_stages[current_stage_index]
                stage_game_config = game_config_for_selfplay.copy() # Start with base game config
                stage_game_config.update(current_stage_config) # Override with stage specifics
                self_play_env = SelfPlay(cfr_trainer=cfr_trainer, game_engine_config=stage_game_config)
                print(f"SelfPlay environment re-initialized with new curriculum stage: {current_stage_config}")
            elif hand_num == 1 and current_stage_index == 0 and not hasattr(self_play_env, '_curriculum_applied_initially'):
                # This ensures the first stage is applied if not done above
                # (The above logic might miss stage 0 if hands_per_stage makes new_stage_index > 0 on first few hands)
                # Simpler: the initial setup of self_play_env before loop already handles stage 0.
                pass # Initial stage already applied before loop if curriculum_stages is not empty

        try:
            _ = self_play_env.play_hand_for_training()
            print(f"Hand {hand_num} completed.")
        except Exception as e:
            print(f"Error during hand {hand_num}: {e}")
            import traceback
            traceback.print_exc()
            # Decide if training should continue or break on error

        if hand_num % save_model_every_n_hands == 0:
            print(f"\n--- Saving model at hand {hand_num} ---")
            try:
                cfr_trainer.save_model()
                print("Model saved successfully.")
            except Exception as e:
                print(f"Error saving model at hand {hand_num}: {e}")
    
    # Final save after the loop
    print("\n--- Training session finished ---")
    print("Saving final model...")
    try:
        cfr_trainer.save_model()
        print("Final model saved successfully.")
    except Exception as e:
        print(f"Error saving final model: {e}")

    print("\n--- Training Complete ---")

if __name__ == '__main__':
    main()
