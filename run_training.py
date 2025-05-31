import yaml
import os # For path manipulation if needed, e.g. for robust config loading

# Assuming the script is run from the project root,
# and trainers, self_play, etc., are packages in that root.
from trainers.ai_cfr_trainer import AICFRTrainer
from self_play.self_play import SelfPlay

# Configuration Loading
# Robustly locate config.yaml assuming it's in the project root
CONFIG_FILE_PATH = "config.yaml" 
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


def main():
    print("--- Starting Poker AI Training Session ---")

    # Load configuration
    config = load_configuration(CONFIG_FILE_PATH)

    # Extract configurations with defaults
    # model_config is implicitly used by AICFRTrainer via its own global config load.
    # We don't directly use model_config here, but AICFRTrainer does.
    
    game_engine_config = config.get('game_engine', {})
    training_params = config.get('training', {})

    # Training Parameters
    num_training_hands = training_params.get('num_training_hands', 1000)
    save_model_every_n_hands = training_params.get('save_model_every_n_hands', 100)
    
    # Game Engine Parameters for SelfPlay
    num_players = game_engine_config.get('num_players', 2)
    starting_stack = game_engine_config.get('starting_stack', 1000)
    big_blind = game_engine_config.get('big_blind', 10)
    small_blind = game_engine_config.get('small_blind', 5)

    print("\n--- Configuration ---")
    print(f"Total training hands: {num_training_hands}")
    print(f"Save model every: {save_model_every_n_hands} hands")
    print(f"Number of players: {num_players}")
    print(f"Starting stack: {starting_stack}")
    print(f"Blinds: SB={small_blind}, BB={big_blind}")
    # Note: AICFRTrainer also loads config.yaml internally for its model parameters.
    # Ensure d_raw_feature is present in config.yaml for TransformerAverageStrategy if not using defaults.

    # Initialization
    print("\n--- Initializing Components ---")
    game_config_for_selfplay = {
        'num_players': num_players,
        'starting_stack': starting_stack,
        'big_blind': big_blind,
        'small_blind': small_blind
        # Note: player_strategies are handled by SelfPlay using DummyStrategy
    }

    try:
        cfr_trainer = AICFRTrainer() # Loads its own model config from 'config.yaml'
        print("AICFRTrainer initialized.")
    except Exception as e:
        print(f"Error initializing AICFRTrainer: {e}")
        print("Please ensure 'config.yaml' is present and correctly formatted, especially the 'model' section.")
        return # Exit if trainer fails to initialize

    try:
        self_play_env = SelfPlay(cfr_trainer=cfr_trainer, game_engine_config=game_config_for_selfplay)
        print("SelfPlay environment initialized.")
    except Exception as e:
        print(f"Error initializing SelfPlay environment: {e}")
        return # Exit if self-play fails

    # Training Loop
    print("\n--- Starting Training Loop ---")
    for hand_num in range(1, num_training_hands + 1):
        print(f"\n--- Training Hand {hand_num}/{num_training_hands} ---")
        try:
            # The play_hand_for_training method now collects data and calls cfr_trainer.train internally
            _ = self_play_env.play_hand_for_training() 
            # The returned training_data could be used for other logging or analysis here if needed.
            print(f"Hand {hand_num} completed.")
        except Exception as e:
            print(f"Error during hand {hand_num}: {e}")
            # Decide if training should continue or break on error
            # For now, print error and continue to next hand
            import traceback
            traceback.print_exc()


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
