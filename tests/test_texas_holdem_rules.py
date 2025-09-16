import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.texas_holdem import TexasHoldem, TexasHoldemRules  # noqa: E402


def test_post_blinds() -> None:
    rules = TexasHoldemRules(num_players=2, starting_stack=100)
    rules.small_blind = 5
    rules.big_blind = 10
    rules.post_blinds()
    assert rules.pot == 15
    assert rules.player_chips == [90, 95]
    assert rules.bets == [10, 5]
    assert rules.current_bet == 10
    assert rules.total_bets_this_hand == [10, 5]


def test_showdown_awards_blinds() -> None:
    game = TexasHoldem(num_players=2, starting_stack=100, verbose=False)
    game.rules.small_blind = 5
    game.rules.big_blind = 10
    game.initialize_game()

    # No additional betting: both players check/call to proceed directly to showdown.
    for _ in range(2):
        current = game.rules.current_player
        valid = game.get_valid_actions(current)
        if "call" in valid:
            game.process_action(current, "call")
        elif "check" in valid:
            game.process_action(current, "check")
        game.rules.advance_turn()

    winnings = game.perform_showdown()
    assert winnings  # pot must be awarded to at least one player
    assert sum(winnings.values()) == game.rules.pot


def test_bet_and_call() -> None:
    rules = TexasHoldemRules(num_players=2, starting_stack=100)
    rules.small_blind = 5
    rules.big_blind = 10
    rules.post_blinds()
    rules.bet(0, 40)  # player 0 raises to 40
    assert rules.current_bet == 40
    assert rules.player_chips[0] == 60
    assert rules.pot == 45  # 15 blinds + 30 raise diff
    rules.bet(1, 40)  # player 1 calls to 40
    assert rules.player_chips[1] == 60
    assert rules.pot == 80


def test_rotate_dealer() -> None:
    rules = TexasHoldemRules(num_players=3, starting_stack=50)
    assert rules.dealer_button == 0
    rules.rotate_dealer()
    assert rules.dealer_button == 1
    rules.rotate_dealer()
    assert rules.dealer_button == 2
    rules.rotate_dealer()
    assert rules.dealer_button == 0


def test_min_raise_amount_follows_wsop_rules() -> None:
    game = TexasHoldem(num_players=2, starting_stack=100, verbose=False)
    game.rules.small_blind = 5
    game.rules.big_blind = 10
    game.initialize_game()

    assert game.get_min_raise_amount(0) == 10
    game.process_action(0, "raise", raise_amount=10)
    assert game.get_min_raise_amount(1) == 10
