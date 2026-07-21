import os
import sys
from types import MethodType

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.texas_holdem import TexasHoldem  # noqa: E402
from poker_ai.rules.texas_holdem import TexasHoldemRules  # noqa: E402


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


def test_post_blinds_skips_eliminated_players() -> None:
    rules = TexasHoldemRules(num_players=4, starting_stack=100)
    rules.small_blind = 5
    rules.big_blind = 10
    rules.dealer_button = 0
    rules.player_chips = [100, 0, 0, 50]
    rules.active_players = [True, False, False, True]
    rules.bets = [0, 0, 0, 0]
    rules.total_bets_this_hand = [0, 0, 0, 0]

    structured_actions = rules.post_blinds()

    assert rules.bets == [10, 0, 0, 5]
    assert rules.player_chips == [90, 0, 0, 45]
    assert rules.pot == 15
    assert rules.current_bet == 10
    assert rules.previous_raise_amount == 10
    assert rules.current_player == 3
    assert structured_actions == [
        ("3", ("bet", 5)),
        ("0", ("bet", 10)),
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


def test_hand_strength_heuristic_fallback_for_short_hands() -> None:
    """When fewer than 5 cards are available (e.g. pre-flop showdown),
    ``_hand_strength`` must still return a deterministic, order-preserving
    score rather than raising.  See commit history for the regression that
    replaced the original heuristic with an ``eval7``-only fallback.
    """
    game = TexasHoldem(num_players=2, starting_stack=100, verbose=False)
    # Two-hole-card, zero-board scenario: treys cannot evaluate this, eval7
    # may be absent, so the heuristic path must handle it.
    score_ah, _, rc_ah = game._hand_strength(["Ah", "Kd"], [])
    score_2c, _, rc_2c = game._hand_strength(["2c", "3d"], [])
    # Higher cards should produce a lower (better) score under the treys
    # "lower is better" convention.
    assert score_ah < score_2c
    # Rank class should be populated (treys: 9 == high card, 8 == pair).
    assert rc_ah in (8, 9, 10)
    assert rc_2c in (8, 9, 10)
    # Pair detection: two kings should be classified as a pair (rank_class 8).
    score_pair, _, rc_pair = game._hand_strength(["Kh", "Kd"], [])
    assert rc_pair == 8
    # Pairs must beat high-card hands even when the high-card has a higher top rank.
    score_aa, _, _ = game._hand_strength(["Ah", "Ad"], [])
    score_ak, _, _ = game._hand_strength(["As", "Kc"], [])
    assert score_aa < score_ak  # pair of aces beats ace-high


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
    game.process_action(0, "raise", raise_amount=20)
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
