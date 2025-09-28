import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.texas_holdem import TexasHoldem


def test_engine_initialization():
    game = TexasHoldem(num_players=2, starting_stack=100, verbose=False)
    game.initialize_game()
    assert len(game.rules.hands[0]) == 2
    assert len(game.rules.hands[1]) == 2
    assert game.rules.pot == game.rules.small_blind + game.rules.big_blind


def test_cash_rebuy_and_cashout():
    game = TexasHoldem(num_players=2, starting_stack=100, verbose=False)
    assert game.cash_table is not None
    table = game.cash_table
    # Simulate a bust and rebuy
    game.rules.player_chips[0] = 0
    table.players[0].stack = 0.0
    added = game.rebuy_to_target(0, target_bb=5)
    assert added > 0
    assert int(round(table.players[0].stack)) == game.rules.player_chips[0]

    # Cash the player out and ensure they leave the table
    payout = game.cash_out_player(0)
    assert payout > 0
    assert not table.players[0].seated
