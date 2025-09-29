#!/usr/bin/env python3
"""Run quick algorithm diagnostics with detailed logging."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = _PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from poker_ai.cli.train import initialize_trainer, load_configuration
from poker_ai.logging_utils import log_configuration_snapshot, log_run_metadata, setup_logging
from poker_ai.selfplay.self_play import SelfPlay


def _diagnose_forward_pass(trainer: Any, logger: logging.Logger) -> dict[str, float]:
    model = getattr(trainer, "model")
    card_projection = getattr(model, "card_projection", None)
    history_projection = getattr(model, "history_projection", None)
    card_dim = getattr(card_projection, "in_features", 32)
    history_dim = getattr(history_projection, "in_features", card_dim)
    seq_len = getattr(model, "max_seq_len", None)
    model_cfg = {}
    if isinstance(getattr(trainer, "config", None), dict):
        model_cfg = trainer.config.get("model", {})
    seq_len = int(model_cfg.get("max_seq_len", seq_len or 16))
    hole = torch.zeros(1, card_dim)
    community = torch.zeros(1, card_dim)
    history = torch.zeros(1, seq_len, history_dim)

    start = time.perf_counter()
    with torch.no_grad():
        _ = trainer.model(hole, community, history)
    duration = time.perf_counter() - start
    logger.info("Forward pass succeeded | duration=%.4fs", duration)
    return {"forward_pass_duration_s": duration}


def _diagnose_training_step(trainer: Any, logger: logging.Logger) -> dict[str, float]:
    model = getattr(trainer, "model")
    card_projection = getattr(model, "card_projection", None)
    history_projection = getattr(model, "history_projection", None)
    card_dim = getattr(card_projection, "in_features", 32)
    history_dim = getattr(history_projection, "in_features", card_dim)
    model_cfg = {}
    if isinstance(getattr(trainer, "config", None), dict):
        model_cfg = trainer.config.get("model", {})
    seq_len = int(model_cfg.get("max_seq_len", getattr(model, "max_seq_len", 16)))
    legal_mask = torch.ones(trainer.num_actions, dtype=torch.bool)
    hole = torch.zeros(card_dim)
    community = torch.zeros(card_dim)
    history = torch.zeros(seq_len, history_dim)
    payoffs = torch.randn(trainer.num_actions)

    start = time.perf_counter()
    loss = trainer.train("diagnostic", hole, community, history, payoffs, mask=legal_mask)
    duration = time.perf_counter() - start
    loss_value = float(loss) if loss is not None else float("nan")
    logger.info(
        "Training step diagnostic | loss=%.6f | duration=%.4fs",
        loss_value,
        duration,
    )
    return {"training_step_loss": loss_value, "training_step_duration_s": duration}


def _diagnose_self_play(trainer: Any, config: dict[str, Any], logger: logging.Logger) -> dict[str, float]:
    game_cfg = {"starting_stack": 50, "big_blind": 1, "small_blind": 1, "min_players": 2, "max_players": 2}
    training_cfg = dict(config.get("training", {}))
    training_cfg["min_buffer_before_train"] = min(
        int(training_cfg.get("min_buffer_before_train", 16)),
        16,
    )

    env = SelfPlay(cfr_trainer=trainer, game_engine_config=game_cfg, training_config=training_cfg)
    start = time.perf_counter()
    buffer = env.play_hand_for_training(iteration=1)
    duration = time.perf_counter() - start
    size = len(buffer)
    logger.info(
        "Self-play diagnostic | entries=%s | duration=%.4fs | threshold=%s",
        size,
        duration,
        env.min_buffer_before_train,
    )
    return {"self_play_duration_s": duration, "replay_buffer_size": size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick diagnostics for Poker AI components")
    parser.add_argument("--config", default=None, help="Path to configuration file")
    parser.add_argument(
        "--algorithm",
        default="ai_cfr",
        choices=["ai_cfr", "deep_cfr", "single_network"],
        help="Trainer algorithm to initialize",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON diagnostics results",
    )
    parser.add_argument(
        "--include-self-play",
        action="store_true",
        help="Run a lightweight self-play traversal (may take several seconds)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_configuration(args.config)
    run_id = f"diagnostics-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    setup_logging(config.get("logging"), component="diagnostics", run_id=run_id)
    logger = logging.getLogger("poker_ai.diagnostics")

    log_run_metadata(config=config, extra_context={"command": "diagnostics", "run_id": run_id})
    log_configuration_snapshot(config, logger=logger)
    logging.getLogger("poker_ai.engine").setLevel(logging.WARNING)
    logging.getLogger("poker_ai.engine.texas_holdem").setLevel(logging.WARNING)

    try:
        trainer = initialize_trainer(args.algorithm, config, device="cpu")
    except Exception as exc:
        logger.exception("Failed to initialize trainer: %s", exc)
        return 1

    results: dict[str, Any] = {"algorithm": args.algorithm, "run_id": run_id}

    try:
        results.update(_diagnose_forward_pass(trainer, logger))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Forward pass diagnostic failed: %s", exc)
        results["forward_pass_error"] = str(exc)

    try:
        results.update(_diagnose_training_step(trainer, logger))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Training step diagnostic failed: %s", exc)
        results["training_step_error"] = str(exc)

    if args.include_self_play:
        try:
            results.update(_diagnose_self_play(trainer, config, logger))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Self-play diagnostic failed: %s", exc)
            results["self_play_error"] = str(exc)
    else:
        logger.info("Skipping self-play diagnostic (enable with --include-self-play)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        logger.info("Wrote diagnostics report to %s", output_path)

    logger.info("Diagnostics complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
