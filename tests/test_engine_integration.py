import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.texas_holdem import TexasHoldem


def test_engine_initialization() -> None:
    game = TexasHoldem(num_players=2, starting_stack=100, verbose=False)
    game.initialize_game()
    assert len(game.rules.hands[0]) == 2
    assert len(game.rules.hands[1]) == 2
    assert game.rules.pot == game.rules.small_blind + game.rules.big_blind

