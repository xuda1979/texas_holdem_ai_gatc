"""Command-line entry for basic training loop using self-play."""

import argparse

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
    self_play_env = SelfPlay(cfr_trainer=trainer, game_engine_config=game_cfg)

    for hand_idx in range(1, args.num_hands + 1):
        self_play_env.play_hand_for_training()
        if args.save_interval and hand_idx % args.save_interval == 0:
            trainer.save_model()
    trainer.save_model()


if __name__ == "__main__":
    main()
