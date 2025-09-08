"""Main command-line interface for training models via self-play."""

import argparse
import os  # For path manipulation if needed, e.g. for robust config loading
import time

import torch

from poker_ai.evaluation.performance_analysis import ModelPerformanceAnalyzer

# Assuming the script is run from the project root,
# and trainers, self_play, etc., are packages in that root.
from poker_ai.selfplay.self_play import SelfPlay

# Configuration Loading
# Robustly locate config.yaml assuming it's in the project root
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
# If script is not in root, adjust path:
# CONFIG_FILE_PATH = os.path.join(
#     os.path.dirname(__file__), "config.yaml"
# )  # If config is with script
# Or an absolute path, or environment variable. For now, assume it's in CWD.


def load_configuration(config_path: str) -> dict:
    """Loads YAML configuration from the given path.

    Falls back to an empty configuration if PyYAML is missing or the file cannot
    be parsed so that training can still run with default values."""
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML is not installed. Using default configurations.")
        return {}

    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            print(f"Warning: {config_path} is empty or invalid. Using default configurations.")
            return {}
        return config_data
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default configurations.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing {config_path}: {e}. Using default configurations.")
        return {}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Run Poker AI training session")
    parser.add_argument(
        "--num-hands", type=int, help="Number of hands to simulate (overrides config)"
    )
    parser.add_argument("--save-model-every", type=int, help="Save model every N hands")
    parser.add_argument(
        "--save-minutes", type=int, help="Save model every N minutes (overrides config)"
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        help="Save model every N samples/hands for performance analysis",
    )
    parser.add_argument(
        "--algorithm",
        default="deep_cfr",
        choices=["ai_cfr", "deep_cfr", "single_network"],
        help="Training algorithm to use",
    )
    parser.add_argument(
        "--config", default=CONFIG_FILE_PATH, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help=(
            "Computation device. Defaults to CUDA if available, then CPU. "
            "Use --npu for NPU support."
        ),
    )
    parser.add_argument("--npu", action="store_true", help="Enable training on all available NPUs.")
    return parser.parse_args()


def initialize_trainer(
    algorithm: str, config: dict, device: str, use_all_npus: bool = False
) -> object:
    """Return a trainer instance based on selected algorithm."""
    trainer: object | None = None
    if algorithm == "ai_cfr":
        from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer

        trainer = AICFRTrainer(device=device)
    elif algorithm == "deep_cfr":
        from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer

        model_cfg = config.get("model", {})
        # Default to 18 raw features so that card one-hot encodings fit even if
        # the configuration file cannot be loaded (e.g. when PyYAML is not
        # installed).  Using a smaller default previously resulted in repeated
        # warnings from ``prepare_transformer_input``.
        d_raw = model_cfg.get("d_raw_feature", 18)
        hidden = model_cfg.get("hidden_dim", 128)
        num_actions = model_cfg.get("num_actions", 10)
        lr = model_cfg.get("learning_rate", 1e-3)
        trainer = DeepCFRTrainer(d_raw, hidden, num_actions, learning_rate=lr, device=device)
    elif algorithm == "single_network":
        from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer

        model_cfg = config.get("model", {})
        # Match the default described above for Deep CFR to ensure consistent
        # feature dimensions across training approaches.
        d_raw = model_cfg.get("d_raw_feature", 18)
        hidden = model_cfg.get("hidden_dim", 128)
        num_actions = model_cfg.get("num_actions", 10)
        lr = model_cfg.get("learning_rate", 1e-3)
        trainer = SingleNetworkCFRTrainer(d_raw, hidden, num_actions, lr, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if use_all_npus:
        model_to_wrap = None
        if hasattr(trainer, "advantage_net"):
            model_to_wrap = trainer.advantage_net
        elif hasattr(trainer, "model"):
            model_to_wrap = trainer.model

        if model_to_wrap:
            print("Wrapping model with DataParallel for multi-NPU training.")
            # The model is already on the correct device from the trainer's __init__
            wrapped_model = torch.nn.DataParallel(model_to_wrap)
            if hasattr(trainer, "advantage_net"):
                trainer.advantage_net = wrapped_model
            elif hasattr(trainer, "model"):
                trainer.model = wrapped_model
        else:
            print("Warning: Could not find model to wrap for DataParallel.")

    return trainer


def main() -> None:  # noqa: C901
    print("--- Starting Poker AI Training Session ---")

    args = parse_args()
    device = None
    use_all_npus = False

    if args.npu:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = "npu"
            npu_count = torch.npu.device_count()
            if npu_count > 1:
                use_all_npus = True
                print(f"Multi-NPU training enabled. Found {npu_count} NPUs.")
            else:
                print("NPU training enabled. Found 1 NPU.")
        else:
            print(
                "Warning: --npu flag was specified, but no NPU devices are available. "
                "Falling back to CPU."
            )
            device = "cpu"
    elif args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Load configuration
    config = load_configuration(args.config)

    # Extract configurations with defaults
    # model_config is implicitly used by AICFRTrainer via its own global config load.
    # We don't directly use model_config here, but AICFRTrainer does.

    game_engine_config = config.get("game_engine", {})
    training_params = config.get("training", {})
    # curriculum stages are currently unused but left here for future expansion

    # Training Parameters with CLI overrides
    # In MCCFR, each "hand" is one full traversal, which is one iteration.
    num_iterations = (
        args.num_hands
        if args.num_hands is not None
        else training_params.get("num_training_hands", 10000)
    )
    save_model_every_samples = (
        args.save_samples
        if args.save_samples is not None
        else training_params.get("save_model_every_samples", 100000)
    )
    save_model_every_n_hands = (
        args.save_model_every
        if args.save_model_every is not None
        else training_params.get("save_model_every_n_hands", 0)
    )
    save_model_every_minutes = (
        args.save_minutes
        if args.save_minutes is not None
        else training_params.get("save_model_every_minutes", 10)
    )

    # Game Engine Parameters for SelfPlay
    min_players = game_engine_config.get("min_players", 2)
    max_players = game_engine_config.get("max_players", 10)
    starting_stack = game_engine_config.get("starting_stack", 1000)
    big_blind = game_engine_config.get("big_blind", 10)
    small_blind = game_engine_config.get("small_blind", 5)

    print("\n--- Configuration ---")
    print(f"Total training iterations: {num_iterations}")
    print(f"Save model every: {save_model_every_samples} samples")
    print(f"Players per hand: random {min_players}-{max_players}")
    print(f"Starting stack: {starting_stack}")
    print(f"Blinds: SB={small_blind}, BB={big_blind}")
    print(f"Using device: {device}")

    # Initialization
    print("\n--- Initializing Components ---")
    game_config_for_selfplay = {
        "starting_stack": starting_stack,
        "big_blind": big_blind,
        "small_blind": small_blind,
        "min_players": min_players,
        "max_players": max_players,
    }

    try:
        cfr_trainer = initialize_trainer(args.algorithm, config, device, use_all_npus=use_all_npus)
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
    analyzer = ModelPerformanceAnalyzer(
        models_dir="models",
        save_every_samples=save_model_every_samples,
        tournament_threshold=10,
        device=device,
    )
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

        analyzer.on_iteration_end(cfr_trainer, iteration)
        should_save = False
        current_time = time.time()
        if (
            save_model_every_minutes > 0
            and current_time - last_save_time >= save_model_every_minutes * 60
        ):
            last_save_time = current_time
            should_save = True
        if save_model_every_n_hands > 0 and iteration % save_model_every_n_hands == 0:
            should_save = True
        if should_save:
            os.makedirs("models", exist_ok=True)
            try:
                cfr_trainer.save_model(f"models/{args.algorithm}_iter_{iteration}.pth")
                print("Model saved during training.")
            except Exception as e:
                print(f"Error saving model during training: {e}")

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


if __name__ == "__main__":
    main()
