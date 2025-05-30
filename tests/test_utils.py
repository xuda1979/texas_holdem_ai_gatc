import unittest
import sys
import os

# Adjust the Python path to include the root directory of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.action_mapping import get_action_from_index
# from utils.betting_system import get_betting_options # Function does not exist in the file
from utils.state_representation import _encode_card

# Mock GameState for testing get_action_from_index,
# similar to the one in action_mapping.py's main block
class MockGameStateForActionMapping:
    def __init__(self, pot, current_bet):
        self.pot = pot
        self.current_bet = current_bet

class TestActionMapping(unittest.TestCase):
    def test_get_action_from_index(self):
        player_stack = 200

        # Scenario 1: No current bet (can check or bet/raise)
        gs_no_bet = MockGameStateForActionMapping(pot=100, current_bet=0)
        self.assertEqual(get_action_from_index(0, gs_no_bet, player_stack), ('fold', None))
        self.assertEqual(get_action_from_index(1, gs_no_bet, player_stack), ('check', None))
        self.assertEqual(get_action_from_index(2, gs_no_bet, player_stack), ('call', 0))

        self.assertEqual(get_action_from_index(3, gs_no_bet, player_stack), ('raise', 25))
        self.assertEqual(get_action_from_index(4, gs_no_bet, player_stack), ('raise', 50))
        self.assertEqual(get_action_from_index(5, gs_no_bet, player_stack), ('raise', 75))
        self.assertEqual(get_action_from_index(6, gs_no_bet, player_stack), ('raise', 100))
        self.assertEqual(get_action_from_index(7, gs_no_bet, player_stack), ('raise', 150))
        self.assertEqual(get_action_from_index(8, gs_no_bet, player_stack), ('raise', 200)) # Capped at stack
        self.assertEqual(get_action_from_index(9, gs_no_bet, player_stack), ('raise', 200)) # All-in

        # Scenario 2: Facing a bet
        gs_bet_exists = MockGameStateForActionMapping(pot=150, current_bet=50)
        player_stack_facing_bet = 200
        self.assertEqual(get_action_from_index(0, gs_bet_exists, player_stack_facing_bet), ('fold', None))
        self.assertEqual(get_action_from_index(1, gs_bet_exists, player_stack_facing_bet), ('check', None)) # Legality handled by game engine
        self.assertEqual(get_action_from_index(2, gs_bet_exists, player_stack_facing_bet), ('call', 50))
        self.assertEqual(get_action_from_index(3, gs_bet_exists, player_stack_facing_bet), ('raise', int(round(150 * 0.25))))
        self.assertEqual(get_action_from_index(9, gs_bet_exists, player_stack_facing_bet), ('raise', 200))

        gs_small_stack = MockGameStateForActionMapping(pot=100, current_bet=10)
        small_player_stack = 30
        self.assertEqual(get_action_from_index(6, gs_small_stack, small_player_stack), ('raise', 30))
        self.assertEqual(get_action_from_index(9, gs_small_stack, small_player_stack), ('raise', 30))

        gs_zero_pot = MockGameStateForActionMapping(pot=0, current_bet=0)
        self.assertEqual(get_action_from_index(3, gs_zero_pot, player_stack), ('raise', 0))

        gs_raise_all_in = MockGameStateForActionMapping(pot=100, current_bet=50)
        player_stack_for_raise_all_in = 70 # Call is 50, remaining is 20. 100% pot raise is 100. Capped at 70.
        action_str, amount = get_action_from_index(6, gs_raise_all_in, player_stack_for_raise_all_in)
        self.assertEqual(action_str, "raise")
        self.assertEqual(amount, 70)

        with self.assertRaises(ValueError):
            get_action_from_index(10, gs_no_bet, player_stack)


# class TestBettingSystem(unittest.TestCase):
#     def test_get_betting_options(self):
#         big_blind=20
#         options1 = get_betting_options(player_stack=1000, current_bet=0, pot_size=100, min_raise_increment=big_blind, big_blind=big_blind)
#         self.assertTrue(options1['check'])
#         self.assertTrue(options1['fold'])
#         self.assertEqual(options1['min_bet'], big_blind)
#         self.assertEqual(options1['pot_bet'], 100)
#         self.assertEqual(options1['all_in'], 1000)
#         self.assertFalse(options1['call'])
#         self.assertIsNone(options1['min_raise'])
#         self.assertIsNone(options1['pot_raise'])

#         player_stack = 500
#         current_bet_on_table = 50
#         pot_size = 150
#         min_raise_inc = 50

#         options2 = get_betting_options(player_stack, current_bet_on_table, pot_size, min_raise_inc, big_blind)
#         self.assertTrue(options2['fold'])
#         self.assertEqual(options2['call'], 50)
#         self.assertFalse(options2['check'])
#         self.assertIsNone(options2['min_bet'])
#         self.assertIsNone(options2['pot_bet'])
#         self.assertEqual(options2['min_raise'], 100)
#         self.assertEqual(options2['pot_raise'], 250)
#         self.assertEqual(options2['all_in'], 500)

#         player_stack_small = 80
#         options3 = get_betting_options(player_stack_small, current_bet_on_table, pot_size, min_raise_inc, big_blind)
#         self.assertEqual(options3['call'], 50)
#         self.assertIsNone(options3['min_raise'])
#         self.assertIsNone(options3['pot_raise'])
#         self.assertEqual(options3['all_in'], 80)

#         player_stack_tiny = 30
#         options4 = get_betting_options(player_stack_tiny, current_bet_on_table, pot_size, min_raise_inc, big_blind)
#         self.assertEqual(options4['call'], 30)
#         self.assertIsNone(options4['min_raise'])
#         self.assertIsNone(options4['pot_raise'])
#         self.assertEqual(options4['all_in'], 30)

#         player_stack_v_small = 5
#         options5 = get_betting_options(player_stack_v_small, 0, 10, big_blind, big_blind)
#         self.assertTrue(options5['check'])
#         self.assertEqual(options5['min_bet'], 5)
#         self.assertEqual(options5['pot_bet'], 5)
#         self.assertEqual(options5['all_in'], 5)

#         player_stack_mid = 150
#         options6 = get_betting_options(player_stack_mid, current_bet_on_table, pot_size, min_raise_inc, big_blind)
#         self.assertEqual(options6['call'], 50)
#         self.assertEqual(options6['min_raise'], 100)
#         self.assertIsNone(options6['pot_raise'])
#         self.assertEqual(options6['all_in'], 150)


class TestStateRepresentationHelpers(unittest.TestCase):
    def test_encode_card(self):
        self.assertAlmostEqual(_encode_card('As'), 14.1)
        self.assertAlmostEqual(_encode_card('2c'), 2.4)
        self.assertAlmostEqual(_encode_card('Th'), 10.2)
        self.assertAlmostEqual(_encode_card('Qd'), 12.3)
        with self.assertRaises(ValueError):
            _encode_card('Xy')
        with self.assertRaises(ValueError):
            _encode_card('A')
        with self.assertRaises(ValueError):
            _encode_card('12s')

if __name__ == '__main__':
    unittest.main()
