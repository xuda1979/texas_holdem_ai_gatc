import os
import sys
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.texas_holdem import TexasHoldemRules


def test_post_blinds():
    rules = TexasHoldemRules(num_players=2, starting_stack=100)
    rules.small_blind = 5
    rules.big_blind = 10
    rules.post_blinds()
    assert rules.pot == 15
    assert rules.player_chips == [90, 95]
    assert rules.bets == [10, 5]
    assert rules.current_bet == 10


def test_bet_and_call():
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


def test_rotate_dealer():
    rules = TexasHoldemRules(num_players=3, starting_stack=50)
    assert rules.dealer_button == 0
    rules.rotate_dealer()
    assert rules.dealer_button == 1
    rules.rotate_dealer()
    assert rules.dealer_button == 2
    rules.rotate_dealer()
    assert rules.dealer_button == 0
