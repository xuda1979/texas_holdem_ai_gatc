"""Command-line entry for basic training loop using self-play."""

import argparse
import os
from typing import Any

from poker_ai.selfplay.self_play import SelfPlay
from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple self-play training")
    parser.add_argument("--num-hands", type=int, default=100, help="Number of hands to simulate")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players at the table")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_cfg = {
        "num_players": args.num_players,
        "starting_stack": args.starting_stack,
    }

    trainer = AICFRTrainer()

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
    if args.save_interval:
        training_cfg["save_model_every_n_hands"] = args.save_interval

    self_play_env = SelfPlay(
        cfr_trainer=trainer,
        game_engine_config=game_cfg,
        training_config=training_cfg or None,
    )

    for hand_idx in range(1, args.num_hands + 1):
        self_play_env.play_hand_for_training(hand_idx)
        if args.save_interval and hand_idx % args.save_interval == 0:
            trainer.save_model()
    trainer.save_model()


if __name__ == "__main__":
    main()
