import random

from poker_ai.engine.texas_holdem import TexasHoldem


def test_replay_round_trip():
    seed = 42
    random.seed(seed)
    game1 = TexasHoldem(2, starting_stack=50, verbose=False)
    game1.initialize_game()
    stacks1 = game1.rules.player_chips[:]

    random.seed(seed)
    game2 = TexasHoldem(2, starting_stack=50, verbose=False)
    game2.initialize_game()
    stacks2 = game2.rules.player_chips[:]

    assert stacks1 == stacks2
