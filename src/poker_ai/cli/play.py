# human_vs_ai.py

"""Command line game allowing humans to play against simple AI players."""

import argparse
import sys
from typing import Any, cast

from poker_ai.ai import load_model_strategy
from poker_ai.config import load_config
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import (
    HumanStrategy,
    ModelAIStrategy,
    RandomAIStrategy,
)
from poker_ai.utils.model_paths import find_latest_model_checkpoint


def parse_args() -> argparse.Namespace:
    """Return parsed command line arguments."""
    parser = argparse.ArgumentParser(description="Play Texas Hold'em against simple AI players")
    parser.add_argument("--total-players", type=int, help="Total number of players (2-10)")
    parser.add_argument("--num-humans", type=int, help="Number of human players")
    parser.add_argument("--starting-stack", type=int, help="Starting chip count")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved AdvantageNetwork weights (.pth) to control AI players",
    )
 
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="Number of hands to play before exiting (0 for infinite play)",
    )
 
    parser.add_argument("--config", default=None, help="Path to configuration YAML file")
 
    return parser.parse_args()


def load_model(model_path: str | None) -> ModelAIStrategy | None:
    """Load a saved AI model using :func:`load_model_strategy`."""

    if model_path is None:
        return None

    strategy, _ = load_model_strategy(model_path)
    if isinstance(strategy, ModelAIStrategy):
        print(f"Loaded AI model from {model_path}\n")
        return strategy
    return None


def main() -> None:  # noqa: C901
    args = parse_args()
    print("=== Welcome to Texas Hold'em Poker Simulation ===\n")

    cfg = load_config(args.config)

    model_path = args.model_path
    latest: tuple[str, float] | None = None
    if model_path is None:
        latest = find_latest_model_checkpoint()
        if latest is not None:
            model_path, _ = latest
            print(f"Using latest available AI model at {model_path}\n")
        else:
            print("No saved AI model found. AI players will use random decisions.\n")

    model_strategy = load_model(model_path)
    if model_strategy is None and model_path is not None:
        if args.model_path:
            print(
                f"Failed to load model from {model_path}. "
                "Falling back to random AI players.\n"
            )
        elif latest is not None:
            print(
                f"The detected AI model at {model_path} could not be loaded. "
                "Using random AI players instead.\n"
            )

    total_players = args.total_players
    if total_players is None:
        while True:
            try:
                total_players = int(input("Enter the total number of players (2 to 10): "))
                if 2 <= total_players <= 10:
                    break
                else:
                    print("Total number of players must be between 2 and 10. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 2 and 10.")
    else:
        if not 2 <= total_players <= 10:
            raise ValueError("--total-players must be between 2 and 10")

    num_humans = args.num_humans
    if num_humans is None:
        while True:
            try:
                num_humans = int(
                    input(f"Enter the number of human players (0 to {total_players}): ")
                )
                if 0 <= num_humans <= total_players:
                    break
                else:
                    print(
                        "Number of human players must be between 0 and "
                        f"{total_players}. Please try again."
                    )
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    else:
        if not 0 <= num_humans <= total_players:
            raise ValueError("--num-humans must be between 0 and total players")

    num_ai = total_players - num_humans

    # Choose tournament type and set starting stack
    starting_stack = (
        args.starting_stack
        if args.starting_stack is not None
        else cfg.get("game_engine", {}).get("starting_stack", 10000)
    )
    if starting_stack is None:
        print("\nChoose tournament type:")
        print("1. Standard Tournament (10,000 chips)")
        choice = input("Enter choice (1): ")
        if choice == "1":
            starting_stack = 10000
        else:
            print("Invalid choice. Defaulting to Standard Tournament.")
            starting_stack = 10000

    # Create player strategies
    player_strategies: list[Any] = []
    for _ in range(num_humans):
        player_strategies.append(HumanStrategy())
    for _ in range(num_ai):
        if model_strategy:
            player_strategies.append(model_strategy)
        else:
            player_strategies.append(RandomAIStrategy())

    cash_defaults = cfg.get("cash_game", {})
    base_small_blind = cash_defaults.get(
        "small_blind", cfg.get("game_engine", {}).get("small_blind", 10)
    )
    base_big_blind = cash_defaults.get(
        "big_blind", cfg.get("game_engine", {}).get("big_blind", 20)
    )
    default_buyin_bb = cash_defaults.get("default_buyin_bb")
    if default_buyin_bb is None:
        blind_for_buyin = base_big_blind or 1
        default_buyin_bb = max(1, int(round(starting_stack / float(blind_for_buyin))))

    cash_config = {
        "small_blind": base_small_blind,
        "big_blind": base_big_blind,
        "min_buyin_bb": cash_defaults.get("min_buyin_bb", 40),
        "max_buyin_bb": cash_defaults.get("max_buyin_bb", 100),
        "default_buyin_bb": default_buyin_bb,
        "default_bankroll_buyins": cash_defaults.get("default_bankroll_buyins", 5),
        "min_bankroll_buyins": cash_defaults.get("min_bankroll_buyins", 1),
        "max_bankroll_buyins": cash_defaults.get("max_bankroll_buyins", 5),
        "rake_pct": cash_defaults.get("rake_pct", 0.0),
        "rake_cap": cash_defaults.get("rake_cap", 0.0),
        "no_flop_no_drop": cash_defaults.get("no_flop_no_drop", True),
    }

    # Instantiate the game with chosen starting stack as an initial buy-in
    game = TexasHoldem(
        total_players,
        starting_stack,
        cast(list[Any], player_strategies),
        cash_config=cash_config,
    )

    hands_to_play = args.num_hands
    hands_played = 0

    try:
        while hands_to_play == 0 or hands_played < hands_to_play:
            try:
                game.play_game()
            except RuntimeError as exc:
                print(f"Session halted: {exc}")
                break
            winner_info = getattr(game, "last_winner", None)
            if isinstance(winner_info, list) and winner_info:
                players = ", ".join(f"Player {idx + 1}" for idx in winner_info)
                print(f"Hand Summary: Tie between {players}.")
            elif isinstance(winner_info, int):
                print(f"Hand Summary: Player {winner_info + 1} wins!")
            else:
                print("Hand Summary: No winner determined.")
            print("\n--- Hand Completed ---")
            if game.cash_table is not None:
                table = game.cash_table
                print("Current table status:")
                for pid in range(total_players):
                    player = table.players.get(pid)
                    stack = int(round(game.rules.player_chips[pid]))
                    bankroll = int(round(player.bankroll)) if player else 0
                    status = "Seated" if player and player.seated else "Away"
                    print(
                        f"  Player {pid + 1}: stack={stack} | bankroll={bankroll} | status={status}"
                    )
                # Allow human players to manage their stacks
                for pid in range(num_humans):
                    player = table.players.get(pid)
                    if player is None or not player.seated:
                        continue
                    while True:
                        prompt = (
                            f"Player {pid + 1} action (Enter=skip, 'max'=top up to {table.max_buyin_bb}bb, "
                            "chip amount, or 'leave'): "
                        )
                        choice = input(prompt).strip().lower()
                        if choice == "":
                            break
                        if choice == "leave":
                            payout = game.cash_out_player(pid)
                            print(
                                f"Player {pid + 1} cashes out {int(round(payout))} chips and leaves the table."
                            )
                            break
                        if choice == "max":
                            added = game.rebuy_to_target(pid)
                            if added > 0:
                                print(
                                    f"Player {pid + 1} tops up by {int(round(added))} chips."
                                )
                            else:
                                print("Unable to top up (insufficient bankroll or already at max).")
                            break
                        try:
                            chips = int(choice)
                        except ValueError:
                            print("Invalid input. Provide a number, 'max', or 'leave'.")
                            continue
                        added = game.rebuy_amount(pid, chips)
                        if added > 0:
                            print(
                                f"Player {pid + 1} adds {int(round(added))} chips to their stack."
                            )
                        else:
                            print("No chips added (check bankroll and table limits).")
                        break

                # Automatically rebuy AI players if they are bust but have bankroll
                for pid in range(num_humans, total_players):
                    player = table.players.get(pid)
                    if (
                        player
                        and player.seated
                        and player.stack < table.small_blind
                        and player.bankroll > 0
                    ):
                        added = game.rebuy_to_target(pid)
                        if added > 0:
                            print(
                                f"AI Player {pid + 1} auto-top-ups by {int(round(added))} chips."
                            )

            hands_played += 1
            if hands_to_play != 0 and hands_played >= hands_to_play:
                break

            if game.cash_table is not None:
                remaining = [
                    pid
                    for pid, player in game.cash_table.players.items()
                    if player.seated and player.stack > 0
                ]
                if not remaining:
                    print("No seated players with chips remain. Ending session.")
                    break
            print("Starting next cash hand...\n")
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
        sys.exit()


if __name__ == "__main__":
    main()
