import pytest

from poker_ai.engine.texas_holdem_simple import TexasHoldem


@pytest.mark.parametrize(
    "hand, expected",
    [
        (["T♠", "J♠", "Q♠", "K♠", "A♠"], (10,)),
        (
            [("A", "♠"), ("A", "♥"), ("A", "♦"), ("A", "♣"), ("K", "♠")],
            (8, 12, 11),
        ),
    ],
)
def test_hand_rank_accepts_multiple_card_representations(hand, expected):
    game = TexasHoldem(num_players=2)
    assert game.hand_rank(hand) == expected


def test_evaluate_hand_with_string_cards():
    game = TexasHoldem(num_players=2)
    game.community_cards = ["2♠", "3♠", "4♠", "5♠", "9♦"]
    player_hand = ["6♠", "7♠"]

    # Should evaluate to a seven-high straight flush. The evaluator reports the
    # index of the highest card (0 = deuce), so 7-high corresponds to ``5``.
    assert game.evaluate_hand(player_hand) == (9, 5)
