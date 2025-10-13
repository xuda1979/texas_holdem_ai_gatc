#!/usr/bin/env python3
"""
Thin wrapper that delegates to the real CLIs under `src/poker_ai/cli`.
It also works from an uninstalled checkout by putting `src/` on sys.path.
"""

# Ensure src/ imports and optional deterministic seeding for local runs.
import poker_ai_bootstrap as _pab

_pab.seed_all()  # no-op unless RUN_DETERMINISTIC=1 or SEED is set
del _pab

import argparse
import os
import sys

# Ensure the `src` package directory is importable when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Delegate to poker_ai.cli.train (pass through extra flags)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Delegate to poker_ai.cli.play (pass through extra flags)",
    )
    args, unknown = parser.parse_known_args()

    if args.train:
        # Hand over argument parsing to the training CLI.
        from poker_ai.cli.train import main as train_main

        sys.argv = ["train"] + unknown
        train_main()
        return

    if args.play:
        # Prefer the play CLI; gracefully fall back to a single-hand demo.
        try:
            from poker_ai.cli.play import main as play_main

            sys.argv = ["play"] + unknown
            play_main()
        except Exception:
            from poker_ai.engine.texas_holdem import TexasHoldem

            game = TexasHoldem(num_players=2)
            game.initialize_game()
            game.play_round()
            winner, best_hand = game.determine_winner()
            print(f"The winner is Player {winner} with the hand: {best_hand}")
        return

    # If neither flag was provided, show a brief help.
    parser.print_help()


if __name__ == "__main__":
    main()
