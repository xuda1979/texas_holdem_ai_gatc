import pytest

from poker_ai.engine.texas_holdem import TexasHoldem


@pytest.fixture
def initialized_game():
    game = TexasHoldem(num_players=3, starting_stack=200, verbose=False)
    game.rules.big_blind = 20
    game.rules.small_blind = 10
    game.initialize_game()

    # Play a simple pre-flop sequence to touch mutable fields.
    for _ in range(2):
        current = game.rules.current_player
        game.process_action(current, "call")
        game.rules.advance_turn()
    current = game.rules.current_player
    game.process_action(current, "check")

    game.winner = [0, 2]
    game.last_winner = [1]
    return game


def test_clone_preserves_core_state(initialized_game):
    game = initialized_game
    clone = game.clone()

    assert clone is not game
    assert clone.rules is not game.rules

    # Basic scalar attributes
    assert clone.num_players == game.num_players
    assert clone.starting_stack == game.starting_stack
    assert clone.end_game_early == game.end_game_early
    assert clone.rules.current_player == game.rules.current_player
    assert clone.rules.current_bet == game.rules.current_bet
    assert clone.rules.pot == game.rules.pot

    # Lists should be equal but not shared
    assert clone.rules.deck == game.rules.deck
    assert clone.rules.deck is not game.rules.deck
    assert clone.rules.community_cards == game.rules.community_cards
    assert clone.rules.community_cards is not game.rules.community_cards
    assert clone.rules.bets == game.rules.bets
    assert clone.rules.bets is not game.rules.bets
    assert clone.rules.player_chips == game.rules.player_chips
    assert clone.rules.player_chips is not game.rules.player_chips
    assert clone.rules.total_bets_this_hand == game.rules.total_bets_this_hand
    assert clone.rules.total_bets_this_hand is not game.rules.total_bets_this_hand
    assert clone.rules.betting_history == game.rules.betting_history
    assert clone.rules.betting_history is not game.rules.betting_history
    assert clone.rules.hands == game.rules.hands
    assert all(c_hand is not o_hand for c_hand, o_hand in zip(clone.rules.hands, game.rules.hands))
    assert clone.current_hand_initial_actions == game.current_hand_initial_actions
    assert clone.current_hand_initial_actions is not game.current_hand_initial_actions

    # Winner fields are copied defensively for list-based ties
    assert clone.winner == game.winner
    assert clone.winner is not game.winner
    assert clone.last_winner == game.last_winner
    assert clone.last_winner is not game.last_winner


def test_clone_does_not_mutate_original(initialized_game):
    game = initialized_game
    clone = game.clone()

    original_deck_len = len(game.rules.deck)

    clone.rules.bets[0] += 5
    assert clone.rules.bets[0] != game.rules.bets[0]

    clone.rules.community_cards.append("Ah")
    assert "Ah" not in game.rules.community_cards

    clone.rules.deck.pop()
    assert len(clone.rules.deck) == original_deck_len - 1
    assert len(game.rules.deck) == original_deck_len

    clone.winner.append(1)
    assert game.winner == [0, 2]

    clone.last_winner.append(2)
    assert game.last_winner == [1]
