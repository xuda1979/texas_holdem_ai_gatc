"""Main command-line interface for training models via self-play."""

import argparse
import os
import time
from pathlib import Path
from typing import Any, cast

import torch

from poker_ai.ai.models.transformer import AdvantageNetwork

from poker_ai.config import load_config
from poker_ai.evaluation.performance_analysis import ModelPerformanceAnalyzer
from poker_ai.utils.model_paths import find_latest_model_checkpoint

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
        "--min-buffer-before-train",
        type=int,
        help="Replay buffer size required before triggering a training batch",
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
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Use a TPU via torch_xla for training.",
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
        The computation device identifier (``cpu``, ``cuda``, ``npu`` or ``xla``).
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
        d_raw = int(model_cfg.get("d_raw_feature", 18))
        hidden = int(model_cfg.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM))
        num_actions = int(model_cfg.get("num_actions", 10))
        lr = float(model_cfg.get("learning_rate", 1e-3))
        trainer = DeepCFRTrainer(d_raw, hidden, num_actions, learning_rate=lr, device=device)
    elif algorithm == "single_network":
        from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer

        model_cfg = config.get("model", {})
        # Match the default described above for Deep CFR to ensure consistent
        # feature dimensions across training approaches.
        d_raw = int(model_cfg.get("d_raw_feature", 18))
        hidden = int(model_cfg.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM))
        num_actions = int(model_cfg.get("num_actions", 10))
        lr = float(model_cfg.get("learning_rate", 1e-3))
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


def _safe_resolve(path: str) -> str:
    """Return a best-effort resolved filesystem path."""

    try:
        return str(Path(path).resolve())
    except OSError:
        return path


def _resume_trainer_from_checkpoint(
    trainer: object, training_cfg: dict[str, Any], model_cfg: dict[str, Any]
) -> str | None:
    """Attempt to load the newest checkpoint so training can resume."""

    loader = getattr(trainer, "load_model", None)
    if not callable(loader):
        return None

    candidates: dict[str, float] = {}

    raw_path = training_cfg.get("save_model_path")
    if isinstance(raw_path, (str, os.PathLike)):
        try:
            path_obj = Path(raw_path)
            if path_obj.exists():
                resolved = str(path_obj.resolve())
                candidates[resolved] = path_obj.stat().st_mtime
        except OSError:
            pass

    directory_hint = model_cfg.get("directory")
    prefix_hint = model_cfg.get("filename_prefix")
    latest = find_latest_model_checkpoint(directory=directory_hint, prefix=prefix_hint)
    if latest is not None:
        latest_path, latest_mtime = latest
        resolved_latest = _safe_resolve(latest_path)
        candidates[resolved_latest] = latest_mtime

    if not candidates:
        return None

    checkpoint_path, _ = max(candidates.items(), key=lambda item: item[1])

    try:
        result = loader(checkpoint_path)
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found; starting from scratch.")
        return None
    except Exception as exc:
        print(f"Failed to load checkpoint {checkpoint_path}: {exc}")
        return None

    if result is False:
        print(f"Trainer declined to load checkpoint {checkpoint_path}; starting fresh.")
        return None

    print(f"Resumed trainer from checkpoint: {checkpoint_path}")
    return checkpoint_path


def _update_latest_model_checkpoint(trainer: object, latest_path: str | None) -> None:
    """Persist the latest weights to ``latest_path`` if possible."""

    if not latest_path:
        return

    saver = getattr(trainer, "save_model", None)
    if not callable(saver):
        return

    try:
        saver(latest_path)
    except Exception as exc:
        print(f"Warning: Failed to update latest model checkpoint at {latest_path}: {exc}")
    else:
        print(f"Latest model checkpoint updated: {latest_path}")


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
        "min_buffer_before_train",
    ):
        if isinstance(getattr(args, attr, None), MagicMock):
            setattr(args, attr, None)
    for flag in ("gpus", "npus", "tpu"):
        if isinstance(getattr(args, flag, None), MagicMock):
            setattr(args, flag, False)

    device: str = "cpu"
    use_data_parallel = False
    device_printable = device

    if args.tpu and (args.gpus or args.npus):
        raise ValueError("Cannot specify --tpu with --gpus or --npus.")

    if args.gpus and args.npus:
        raise ValueError("Cannot specify both --gpus and --npus.")

    if args.tpu:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - dependency not installed
            raise RuntimeError(
                "torch_xla is required for TPU training. Install the torch-xla package first."
            ) from exc

        xla_device = xm.xla_device()
        device = "xla"
        device_printable = f"{device} ({xla_device})"
        print(f"TPU training enabled on device {xla_device}.")
    elif args.npus:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = "npu"
            device_printable = device
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
            device_printable = device
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

    analyzer_device = device if device != "xla" else "cpu"
    if device == "xla" and analyzer_device == "cpu":
        print("ModelPerformanceAnalyzer evaluations will run on the CPU while training on TPU.")

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

    min_buffer_override = args.min_buffer_before_train
    if min_buffer_override is not None:
        min_buffer_before_train = int(min_buffer_override)
    else:
        min_buffer_before_train = int(
            training_params.get("min_buffer_before_train", 256)
        )
    training_params["min_buffer_before_train"] = min_buffer_before_train

    model_config = config.get("model", {})
    training_params.setdefault("model_directory", model_config.get("directory"))
    training_params.setdefault("model_filename_prefix", model_config.get("filename_prefix"))

    # Game Engine Parameters for SelfPlay
    min_players = game_engine_config.get("min_players", 2)
    max_players = game_engine_config.get("max_players", 10)
    starting_stack = game_engine_config.get("starting_stack", 1000)
    big_blind = game_engine_config.get("big_blind", 10)
    small_blind = game_engine_config.get("small_blind", 5)

    latest_model_path_raw = training_params.get("save_model_path")
    latest_model_path = (
        os.fspath(latest_model_path_raw)
        if isinstance(latest_model_path_raw, (str, os.PathLike))
        else None
    )

    print("\n--- Configuration ---")
    print(f"Total training iterations: {num_iterations}")
    print(f"Save model every: {save_model_every_samples} iterations")
    if save_model_every_n_hands > 0:
        print(f"Save model every {save_model_every_n_hands} hands")
    if save_model_every_minutes > 0:
        print(f"Save model every {save_model_every_minutes} minutes")
    print(f"Players per hand: random {min_players}-{max_players}")
    print(f"Starting stack: {starting_stack}")
    print(f"Blinds: SB={small_blind}, BB={big_blind}")
    print(f"Using device: {device_printable}")
    print(f"Min buffer before training: {min_buffer_before_train}")

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
                args.algorithm,
                config,
                device,
                use_data_parallel=use_data_parallel,
            ),
        )
        print(f"{args.algorithm} trainer initialized.")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        import traceback

        traceback.print_exc()
        return

    resumed_checkpoint = _resume_trainer_from_checkpoint(
        cfr_trainer, training_params, model_config
    )
    if resumed_checkpoint is not None and latest_model_path:
        if _safe_resolve(latest_model_path) != _safe_resolve(resumed_checkpoint):
            _update_latest_model_checkpoint(cfr_trainer, latest_model_path)

    # The new SelfPlay class for MCCFR doesn't need curriculum learning or complex setup.
    # It's simplified for the core algorithm.
    self_play_env = SelfPlay(
        cfr_trainer=cfr_trainer,
        game_engine_config=game_config_for_selfplay,
        training_config=training_params,
    )

    # Training Loop
    print("\n--- Starting Training Loop ---")
    analyzer = ModelPerformanceAnalyzer(
        models_dir="models",
        save_every_iterations=save_model_every_samples,
        tournament_threshold=10,
        device=analyzer_device,
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
            latest_checkpoint_needs_update = False
            if save_model_every_n_hands > 0 and iteration % save_model_every_n_hands == 0:
                path = f"models/{args.algorithm}_hand_{iteration}.pth"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cfr_trainer.save_model(path)
                print(f"Model saved to {path} at iteration {iteration}")
                latest_checkpoint_needs_update = True

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
                latest_checkpoint_needs_update = True

            analyzer.on_iteration_end(cfr_trainer, iteration=iteration)

            if (
                save_model_every_samples > 0
                and iteration % save_model_every_samples == 0
            ):
                latest_checkpoint_needs_update = True

            if latest_checkpoint_needs_update:
                _update_latest_model_checkpoint(cfr_trainer, latest_model_path)
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
            _update_latest_model_checkpoint(cfr_trainer, latest_model_path)
        except Exception as e:
            print(f"Error saving final model: {e}")

        print("\n--- Training Complete ---")


if __name__ == "__main__":
    main()
