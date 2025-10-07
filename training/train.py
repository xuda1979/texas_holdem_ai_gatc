"""Command-line entry for basic training loop using self-play."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from poker_ai.selfplay.self_play import SelfPlay
from poker_ai.ai import trainers as trainer_module


LOGGER = logging.getLogger(__name__)


def _ensure_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = config.get(key)
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section


def _apply_lightweight_defaults(model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> None:
    """Force a configuration that fits within CPU constraints."""

    model_cfg["hidden_dim"] = min(int(model_cfg.get("hidden_dim", 128)), 128)
    model_cfg["num_layers"] = min(int(model_cfg.get("num_layers", 2)), 2)
    model_cfg["num_heads"] = min(int(model_cfg.get("num_heads", 4)), 4)
    model_cfg.setdefault("learning_rate", 1e-3)
    model_cfg.setdefault("d_raw_feature", 18)
    training_cfg.setdefault("replay_buffer_capacity", 4096)


def _override_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the trainer configuration after applying CLI overrides."""

    config = copy.deepcopy(trainer_module.config)

    model_cfg = _ensure_section(config, "model")
    training_cfg = _ensure_section(config, "training")
    logging_cfg = _ensure_section(config, "logging")

    if args.lightweight:
        _apply_lightweight_defaults(model_cfg, training_cfg)

    if args.hidden_dim is not None:
        model_cfg["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        model_cfg["num_layers"] = args.num_layers
    if args.num_heads is not None:
        model_cfg["num_heads"] = args.num_heads
    if args.learning_rate is not None:
        model_cfg["learning_rate"] = args.learning_rate
    if args.max_seq_len is not None:
        model_cfg["max_seq_len"] = args.max_seq_len
    if args.replay_buffer_capacity is not None:
        training_cfg["replay_buffer_capacity"] = args.replay_buffer_capacity
    if args.save_path:
        training_cfg["save_model_path"] = args.save_path

    if args.log_file:
        logging_cfg["log_file"] = args.log_file
    if args.log_level:
        logging_cfg["level"] = args.log_level

    trainer_module.config = config
    trainer_module.ai_cfr_trainer_module.config = config
    sys.modules["poker_ai.ai.trainers.config"] = config
    return config


def _configure_logging(config: dict[str, Any]) -> str | None:
    logging_cfg = config.get("logging", {}) if isinstance(config, dict) else {}
    log_file = logging_cfg.get("log_file")
    log_level = logging_cfg.get("level", "INFO")
    log_format = logging_cfg.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    level = logging.getLevelName(str(log_level).upper())
    if isinstance(level, str):
        level = logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if isinstance(log_file, str) and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))
        log_file = str(log_path)
    else:
        log_file = None

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Reduce extremely verbose trainer logging unless explicitly requested.
    trainer_logger = logging.getLogger("poker_ai.ai.trainers.ai_cfr_trainer")
    trainer_level = logging_cfg.get("trainer_level")
    if trainer_level is not None:
        resolved_level = logging.getLevelName(str(trainer_level).upper())
        if isinstance(resolved_level, str):
            resolved_level = logging.INFO
        trainer_logger.setLevel(resolved_level)
    else:
        trainer_logger.setLevel(logging.WARNING)

    return log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple self-play training")
    parser.add_argument("--num-hands", type=int, default=100, help="Number of hands to simulate")
    parser.add_argument(
        "--min-players",
        type=int,
        default=2,
        help="Minimum number of players to include in randomly generated hands",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=10,
        help="Maximum number of players to include in randomly generated hands",
    )
    parser.add_argument("--starting-stack", type=int, default=1000, help="Starting chip stack")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help=(
            "Number of hands between intermediate model saves. "
            "A value of 0 disables periodic saving."
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override the model hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override the number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Override the number of attention heads",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the optimiser learning rate",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override the maximum sequence length fed to the transformer",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=None,
        help="Override the replay buffer capacity",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Explicit model save location overriding config defaults",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write detailed training progress to this file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Set logging level (e.g. DEBUG, INFO)",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use CPU-friendly defaults for model size and buffer capacity",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable alternating simulation/training cycles for long-running jobs",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=100.0,
        help=(
            "Target duration in hours when running in continuous mode (default: 100)."
        ),
    )
    parser.add_argument(
        "--samples-per-cycle",
        type=int,
        default=10000,
        help=(
            "Number of simulated hands to generate before each training phase in continuous mode"
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
        help="Batch size used for optimisation steps during continuous mode",
    )
    parser.add_argument(
        "--train-steps-per-cycle",
        type=int,
        default=None,
        help=(
            "Number of optimisation batches to run after each generation cycle in continuous "
            "mode. Defaults to covering the freshly generated samples."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _override_config(args)
    log_file = _configure_logging(config)
    LOGGER.info("Starting training with args: %s", args)

    min_players = max(2, args.min_players)
    max_players = max(min_players, args.max_players)

    game_cfg = {
        "min_players": min_players,
        "max_players": max_players,
        "starting_stack": args.starting_stack,
    }

    trainer = trainer_module.AICFRTrainer()

    training_cfg: dict[str, Any] = {}
    trainer_cfg = getattr(trainer, "config", {})
    if isinstance(trainer_cfg, dict):
        training_section = trainer_cfg.get("training")
        if isinstance(training_section, dict):
            save_path = training_section.get("save_model_path")
            if isinstance(save_path, str) and save_path:
                training_cfg["save_model_path"] = save_path
                if os.path.isfile(save_path):
                    trainer.load_model(save_path)
            log_every = training_section.get("log_every_n_hands")
            if log_every is not None:
                training_cfg["log_every_n_hands"] = log_every
    if args.save_interval:
        training_cfg["save_model_every_n_hands"] = args.save_interval

    training_cfg_for_env = training_cfg or None
    train_during_generation = True
    if args.continuous:
        if training_cfg is None:
            training_cfg = {}
        training_cfg.setdefault("train_during_generation", False)
        training_cfg_for_env = training_cfg
        train_during_generation = False

    self_play_env = SelfPlay(
        cfr_trainer=trainer,
        game_engine_config=game_cfg,
        training_config=training_cfg_for_env,
        train_during_generation=train_during_generation,
    )

    log_every = int(training_cfg.get("log_every_n_hands", 50))
    if args.continuous:
        _run_continuous_training(
            self_play_env,
            trainer,
            duration_hours=args.duration_hours,
            samples_per_cycle=args.samples_per_cycle,
            batch_size=args.train_batch_size,
            train_steps_per_cycle=args.train_steps_per_cycle,
            log_every=log_every,
        )
        if log_file:
            LOGGER.info("Detailed logs written to %s", log_file)
    else:
        for hand_idx in range(1, args.num_hands + 1):
            self_play_env.play_hand_for_training(hand_idx)
            if args.save_interval and hand_idx % args.save_interval == 0:
                trainer.save_model()
            if log_every and hand_idx % log_every == 0:
                LOGGER.info("Completed %d hands", hand_idx)
        trainer.save_model()
        LOGGER.info(
            "Training complete. Model saved to %s", training_cfg.get("save_model_path")
        )
        if log_file:
            LOGGER.info("Detailed logs written to %s", log_file)


def _run_continuous_training(
    self_play_env: SelfPlay,
    trainer: object,
    *,
    duration_hours: float,
    samples_per_cycle: int,
    batch_size: int,
    train_steps_per_cycle: int | None,
    log_every: int,
) -> None:
    """Alternate self-play data generation and training for a fixed duration."""

    if duration_hours <= 0:
        LOGGER.warning("Continuous mode requested with non-positive duration. Exiting early.")
        return

    samples_per_cycle = max(1, int(samples_per_cycle))
    batch_size = max(1, int(batch_size))
    if train_steps_per_cycle is not None:
        train_steps_per_cycle = max(1, int(train_steps_per_cycle))

    start_time = time.monotonic()
    end_time = start_time + float(duration_hours) * 3600.0
    cycle = 0
    total_hands = 0

    while True:
        now = time.monotonic()
        if now >= end_time:
            break

        cycle += 1
        elapsed_hours = (now - start_time) / 3600.0
        LOGGER.info(
            "Starting cycle %s | elapsed %.2f/%.2f hours | target samples %s",
            cycle,
            elapsed_hours,
            duration_hours,
            samples_per_cycle,
        )

        replay_buffer = getattr(trainer, "replay_buffer", None)
        if hasattr(replay_buffer, "clear"):
            replay_buffer.clear()

        cycle_start = time.monotonic()
        hands_this_cycle = 0
        for _ in range(samples_per_cycle):
            total_hands += 1
            hands_this_cycle += 1
            self_play_env.play_hand_for_training(total_hands)
            if log_every and total_hands % log_every == 0:
                LOGGER.info("Generated %d hands so far", total_hands)
            if time.monotonic() >= end_time:
                break

        buffer_len = len(replay_buffer) if hasattr(replay_buffer, "__len__") else 0
        steps = train_steps_per_cycle
        if steps is None:
            steps = max(1, math.ceil(buffer_len / batch_size)) if buffer_len else 1

        losses: list[float] = []
        for step in range(1, steps + 1):
            loss = trainer.train(batch_size=batch_size)
            if loss is not None:
                losses.append(float(loss))
            LOGGER.debug(
                "Cycle %s | training step %s/%s | loss=%s",
                cycle,
                step,
                steps,
                "{:.6f}".format(float(loss)) if loss is not None else "n/a",
            )

        trainer.save_model()
        duration = time.monotonic() - cycle_start
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        LOGGER.info(
            (
                "Cycle %s complete | hands=%s | buffer=%s | train_steps=%s | "
                "mean_loss=%.6f | duration=%.2fs"
            ),
            cycle,
            hands_this_cycle,
            buffer_len,
            steps,
            mean_loss,
            duration,
        )

    total_elapsed = time.monotonic() - start_time
    LOGGER.info(
        "Continuous training finished | cycles=%s | hands=%s | elapsed=%.2f hours",
        cycle,
        total_hands,
        total_elapsed / 3600.0,
    )


if __name__ == "__main__":
    main()
