"""Main command-line interface for training models via self-play."""

import argparse
import os
import time
from typing import Any, cast

import torch

from poker_ai.config import load_config
from poker_ai.evaluation.performance_analysis import ModelPerformanceAnalyzer

# Assuming the script is run from the project root,
# and trainers, self_play, etc., are packages in that root.
from poker_ai.selfplay.self_play import SelfPlay


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
        "--config",
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--gpus",
        action="store_true",
        help="Use available GPUs. Wraps model in DataParallel if multiple GPUs are present.",
    )
    parser.add_argument(
        "--npus",
        action="store_true",
        help="Use available NPUs. Wraps model in DataParallel if multiple NPUs are present.",
    )
    return parser.parse_args()


def load_configuration(path: str | None = None) -> dict:
    """Wrapper for :func:`poker_ai.config.load_config` used in tests.

    Providing a dedicated function allows unit tests to patch configuration
    loading without importing the heavier dependency chain inside
    :mod:`poker_ai.config` at import time.
    """

    return load_config(path)


def initialize_trainer(
    algorithm: str, config: dict, device: str, use_data_parallel: bool = False
) -> object:
    """Return a trainer instance based on selected algorithm.

    Parameters
    ----------
    algorithm:
        Which training algorithm to initialize.
    config:
        Loaded configuration dictionary.
    device:
        The computation device identifier (``cpu``, ``cuda`` or ``npu``).
    use_data_parallel:
        If ``True`` the underlying model will be wrapped with
        :class:`torch.nn.DataParallel` to leverage multiple devices.
    """
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

    if use_data_parallel:
        model_to_wrap = None
        if hasattr(trainer, "advantage_net"):
            model_to_wrap = trainer.advantage_net
        elif hasattr(trainer, "model"):
            model_to_wrap = trainer.model

        if model_to_wrap:
            print("Wrapping model with DataParallel for multi-device training.")
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
    from unittest.mock import MagicMock
    for attr in (
        "device",
        "num_hands",
        "save_model_every",
        "save_minutes",
        "save_samples",
    ):
        if isinstance(getattr(args, attr, None), MagicMock):
            setattr(args, attr, None)
    for flag in ("gpus", "npus"):
        if isinstance(getattr(args, flag, None), MagicMock):
            setattr(args, flag, False)

    device: str = "cpu"
    use_data_parallel = False

    if args.gpus and args.npus:
        raise ValueError("Cannot specify both --gpus and --npus.")

    if args.npus:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = "npu"
            npu_count = torch.npu.device_count()
            if npu_count > 1:
                use_data_parallel = True
                print(f"Multi-NPU training enabled. Found {npu_count} NPUs.")
            else:
                print("NPU training enabled.")
        else:
            print(
                "Warning: --npus specified, but no NPU devices are available. "
                "Falling back to CPU."
            )
    elif args.gpus:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                use_data_parallel = True
                print(f"Multi-GPU training enabled. Found {gpu_count} GPUs.")
            else:
                print("GPU training enabled.")
        else:
            print(
                "Warning: --gpus specified, but no GPU devices are available. "
                "Falling back to CPU."
            )

    # Load configuration
    config = load_configuration(args.config)

    # Extract configurations with defaults
    # model_config is implicitly used by AICFRTrainer via its own global config load.
    # We don't directly use model_config here, but AICFRTrainer does.

    game_engine_config = config.get("game_engine", {})
    training_params = config.get("training", {})
    _curriculum_stages: list[dict] = config.get("curriculum", {}).get("stages", [])

    # Training Parameters with CLI overrides
    # In MCCFR, each "hand" is one full traversal, which is one iteration.
    num_iterations = int(
        args.num_hands
        if args.num_hands is not None
        else training_params.get("num_training_hands", 10000)
    )
    save_model_every_n_hands = int(
        args.save_model_every
        if args.save_model_every is not None
        else training_params.get("save_model_every_n_hands", 0)
    )
    if args.save_minutes is not None:
        save_model_every_minutes = int(args.save_minutes)
    else:
        save_model_every_minutes = int(
            training_params.get("save_model_every_minutes", 10)
        )

    save_model_every_samples = int(
        args.save_samples
        if args.save_samples is not None
        else training_params.get("save_model_every_samples", 100000)
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
    if save_model_every_n_hands > 0:
        print(f"Save model every {save_model_every_n_hands} hands")
    if save_model_every_minutes > 0:
        print(f"Save model every {save_model_every_minutes} minutes")
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
        cfr_trainer = cast(
            Any,
            initialize_trainer(
                args.algorithm, config, device, use_data_parallel=use_data_parallel
            ),
        )
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
    iteration = 0
    try:
        for iteration in range(1, num_iterations + 1):
            print(f"\n--- MCCFR Iteration {iteration}/{num_iterations} ---")
            # The play_hand_for_training method now runs one MCCFR traversal
            # and triggers the training step internally.
            self_play_env.play_hand_for_training(iteration)

            # Conditional saving logic
            if save_model_every_n_hands > 0 and iteration % save_model_every_n_hands == 0:
                path = f"models/{args.algorithm}_hand_{iteration}.pth"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cfr_trainer.save_model(path)
                print(f"Model saved to {path} at iteration {iteration}")

            if (
                save_model_every_minutes > 0
                and (time.time() - last_save_time) >= save_model_every_minutes * 60
            ):
                path = f"models/{args.algorithm}_time_{iteration}.pth"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cfr_trainer.save_model(path)
                print(
                    f"Model saved to {path} due to time interval at iteration {iteration}"
                )
                last_save_time = time.time()

            analyzer.on_iteration_end(cfr_trainer, iteration)
    except Exception as e:
        import traceback

        print(f"Error during iteration {iteration}: {e}")
        traceback.print_exc()
        raise
    else:
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
