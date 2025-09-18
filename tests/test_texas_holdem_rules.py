import os
import sys
from types import MethodType

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


def test_post_blinds_short_stacks() -> None:
    rules = TexasHoldemRules(num_players=2, starting_stack=100)
    rules.small_blind = 5
    rules.big_blind = 10
    rules.player_chips = [7, 3]  # Big blind seat first, then small blind seat
    rules.bets = [0, 0]
    rules.total_bets_this_hand = [0, 0]

    structured_actions = rules.post_blinds()

    assert rules.pot == 10
    assert rules.player_chips == [0, 0]
    assert rules.bets == [7, 3]
    assert rules.total_bets_this_hand == [7, 3]
    assert rules.current_bet == 7
    assert rules.previous_raise_amount == 7
    assert structured_actions == [
        ("1", ("bet", 3)),
        ("0", ("bet", 7)),
    ]


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

    winnings, best_hands = game.perform_showdown()
    assert winnings  # pot must be awarded to at least one player
    assert best_hands  # best hand classes reported for active players
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


def test_perform_showdown_returns_best_hand_classes_for_multiway_tie() -> None:
    game = TexasHoldem(num_players=3, starting_stack=500, verbose=False)
    game.rules.active_players = [True, True, True]
    game.rules.community_cards = ["2h", "3d", "4c", "5s", "6h"]
    game.rules.hands = [
        ["Ah", "Ad"],
        ["Ac", "As"],
        ["Kh", "Kd"],
    ]
    game.rules.total_bets_this_hand = [50, 50, 50]
    game.rules.pot = sum(game.rules.total_bets_this_hand)

    outcomes = {
        tuple(game.rules.hands[0]): (10, [], 1),
        tuple(game.rules.hands[1]): (10, [], 1),
        tuple(game.rules.hands[2]): (10, [], 1),
    }

    def fake_hand_strength(self, hole, board):  # pragma: no cover - simple stub
        return outcomes[tuple(hole)]

    game._hand_strength = MethodType(fake_hand_strength, game)

    winnings, best_hands = game.perform_showdown()
    assert sum(winnings.values()) == game.rules.pot
    assert winnings == {0: 50, 1: 50, 2: 50}
    assert best_hands == {0: 1, 1: 1, 2: 1}


def test_perform_showdown_side_pot_split() -> None:
    game = TexasHoldem(num_players=3, starting_stack=1000, verbose=False)
    game.rules.active_players = [True, True, True]
    game.rules.community_cards = ["2h", "3d", "4c", "5s", "6h"]
    game.rules.hands = [
        ["Ah", "Ad"],
        ["Kc", "Kd"],
        ["Qh", "Qd"],
    ]
    contributions = [50, 100, 200]
    game.rules.total_bets_this_hand = contributions[:]
    game.rules.pot = sum(contributions)

    outcomes = {
        tuple(game.rules.hands[0]): (1, [], 7),
        tuple(game.rules.hands[1]): (5, [], 4),
        tuple(game.rules.hands[2]): (10, [], 2),
    }

    def fake_hand_strength(self, hole, board):  # pragma: no cover - simple stub
        return outcomes[tuple(hole)]

    game._hand_strength = MethodType(fake_hand_strength, game)

    winnings, best_hands = game.perform_showdown()
    assert sum(winnings.values()) == game.rules.pot
    assert winnings == {0: 150, 1: 100, 2: 100}
    assert best_hands == {0: 7, 1: 4, 2: 2}
