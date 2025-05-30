import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import random # For seeding tests if necessary

# Adjust the Python path to include the root directory of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game_engine.deck import Deck
from game_engine.game_state import GameState
from game_engine.player import Player as GameStatePlayer # Player class for GameState
from game_engine.texas_holdem import TexasHoldem, TexasHoldemRules
from game_engine.player import Player as TexasHoldemPlayer # Player class for TexasHoldem strategies


class TestDeck(unittest.TestCase):
    def test_deck_initialization(self):
        deck = Deck()
        self.assertEqual(len(deck.cards), 52, "Deck should have 52 cards initially.")
        self.assertEqual(len(set(str(card) for card in deck.cards)), 52, "All cards in the deck should be unique.")

    def test_deck_shuffle(self):
        deck1 = Deck()
        deck2 = Deck()
        initial_order_deck1 = [str(card) for card in deck1.cards]
        initial_order_deck2 = [str(card) for card in deck2.cards]
        self.assertEqual(initial_order_deck1, initial_order_deck2, "Two newly created decks should be in the same order.")
        deck1.shuffle()
        shuffled_order_deck1 = [str(card) for card in deck1.cards]
        self.assertEqual(len(shuffled_order_deck1), 52, "Shuffled deck should still have 52 cards.")
        self.assertEqual(len(set(shuffled_order_deck1)), 52, "Shuffled deck should still have unique cards.")
        self.assertNotEqual(shuffled_order_deck1, initial_order_deck2,
                            "Shuffled deck order should be different from the initial order.")

    def test_deck_deal(self):
        deck = Deck()
        initial_deck_size = len(deck.cards)
        num_to_deal = 5
        hand = deck.deal(num_to_deal)
        self.assertEqual(len(hand), num_to_deal, f"Should deal {num_to_deal} cards.")
        self.assertEqual(len(deck.cards), initial_deck_size - num_to_deal,
                         f"Deck size should decrease by {num_to_deal} after dealing.")
        remaining_cards = len(deck.cards)
        hand2 = deck.deal(remaining_cards)
        self.assertEqual(len(hand2), remaining_cards, "Should deal all remaining cards.")
        self.assertEqual(len(deck.cards), 0, "Deck should be empty after dealing all cards.")
        with self.assertRaisesRegex(ValueError, "Cannot deal 1 cards. Only 0 cards remaining in the deck."):
            deck.deal(1)
        deck3 = Deck()
        with self.assertRaisesRegex(ValueError, "Cannot deal 53 cards. Only 52 cards remaining in the deck."):
            deck3.deal(53)


class TestGameState(unittest.TestCase):
    def test_gamestate_initialization(self):
        p1 = GameStatePlayer(player_id="p1", stack_size=100)
        p1.set_hand(['Ah', 'Ad'])
        p2 = GameStatePlayer(player_id="p2", stack_size=200)
        p2.set_hand(['Kh', 'Kd'])
        players_data = [p1, p2]
        gs = GameState()
        gs.set_players(players_data)
        gs.community_cards = ['Qh', 'Qd', 'Qs']
        gs.pot = 50
        gs.current_bet = 20
        gs.betting_round = "flop"
        gs.betting_history=[("p1", ("bet", 10)), ("p2", ("raise", 20))]
        self.assertEqual(len(gs.players), 2)
        self.assertEqual(gs.players[0].player_id, "p1")
        self.assertEqual(gs.community_cards, ['Qh', 'Qd', 'Qs'])
        self.assertEqual(gs.pot, 50)
        self.assertEqual(gs.current_bet, 20)
        self.assertEqual(gs.betting_round, "flop")
        self.assertEqual(len(gs.betting_history), 2)

    def test_gamestate_record_action(self):
        p1 = GameStatePlayer(player_id="p1", stack_size=100)
        players_data = [p1]
        gs = GameState()
        gs.set_players(players_data)
        self.assertEqual(len(gs.betting_history), 0)
        gs.record_action("p1", ("bet", 50))
        self.assertEqual(len(gs.betting_history), 1)
        self.assertEqual(gs.betting_history[0], ("p1", ("bet", 50)))
        gs.record_action("p1", ("fold", None))
        self.assertEqual(len(gs.betting_history), 2)
        self.assertEqual(gs.betting_history[1], ("p1", ("fold", None)))


class TestTexasHoldem(unittest.TestCase):
    def setUp(self):
        self.dummy_strategies = [MagicMock(spec=TexasHoldemPlayer) for _ in range(2)]
        for i, strategy in enumerate(self.dummy_strategies):
            strategy.player_id = f"player_{i}"
        self.game_config = {
            'num_players': 2,
            'starting_stack': 1000,
            'player_strategies': self.dummy_strategies,
            'big_blind': 10,
            'small_blind': 5
        }
        self.game = TexasHoldem(**self.game_config)

    def test_initialization_default(self):
        # Test state before initialize_game() is called by a game loop or test
        self.assertEqual(self.game.rules.num_players, 2)
        self.assertEqual(len(self.game.rules.player_chips), 2)
        self.assertEqual(self.game.rules.player_chips[0], self.game_config['starting_stack'])
        self.assertEqual(self.game.rules.player_chips[1], self.game_config['starting_stack'])
        self.assertEqual(self.game.rules.big_blind, 10)
        self.assertEqual(self.game.rules.small_blind, 5)
        self.assertIsNotNone(self.game.rules.deck, "Deck object should be in rules")
        initial_deck_card_count = len(self.game.rules.deck) # Should be 52

        self.game.initialize_game() # Now post blinds, deal cards
        self.assertEqual(len(self.game.rules.deck), initial_deck_card_count - (2*2), "Deck size after hole cards dealt")
        # For 2 players, dealer=0. SB is P1 (index 1), BB is P0 (index 0).
        self.assertEqual(self.game.rules.player_chips[1], self.game_config['starting_stack'] - self.game_config['small_blind'])
        self.assertEqual(self.game.rules.player_chips[0], self.game_config['starting_stack'] - self.game_config['big_blind'])

    def test_initialization_custom_players(self):
        strategies = [MagicMock(spec=TexasHoldemPlayer) for _ in range(3)]
        for i, strategy in enumerate(strategies):
            strategy.player_id = f"player_{i}"
        config = {**self.game_config, 'num_players': 3, 'player_strategies': strategies}
        game = TexasHoldem(**config)
        game.initialize_game()
        self.assertEqual(game.rules.num_players, 3)
        self.assertEqual(len(game.rules.player_chips), 3)

        dealer = game.rules.dealer_button # Default is 0 for a new game instance
        sb_idx = (dealer + 1) % 3
        bb_idx = (dealer + 2) % 3
        expected_chips = [self.game_config['starting_stack']] * 3
        expected_chips[sb_idx] -= self.game_config['small_blind']
        expected_chips[bb_idx] -= self.game_config['big_blind']
        self.assertEqual(list(game.rules.player_chips), expected_chips)
        self.assertEqual(len(game.rules.deck), 52 - (3*2))

    def test_deal_hole_cards(self):
        self.game.initialize_game()
        initial_deck_size = 52
        self.assertEqual(len(self.game.rules.hands), self.game.rules.num_players)
        for hand in self.game.rules.hands:
            self.assertEqual(len(hand), 2)
            self.assertIsInstance(hand[0], str)
            self.assertIsInstance(hand[1], str)
        expected_deck_size = initial_deck_size - (self.game.rules.num_players * 2)
        self.assertEqual(len(self.game.rules.deck), expected_deck_size)

    def test_deal_community_cards(self):
        self.game.initialize_game()
        num_players = self.game_config['num_players']

        self.game.play_stage('flop')
        self.assertEqual(len(self.game.rules.community_cards), 3)
        for card_str in self.game.rules.community_cards:
            self.assertIsInstance(card_str, str)
        self.assertEqual(len(self.game.rules.deck), 52 - (num_players*2) - 1 - 3) # 1 burn, 3 flop

        self.game.play_stage('turn')
        self.assertEqual(len(self.game.rules.community_cards), 4)
        self.assertIsInstance(self.game.rules.community_cards[3], str)
        self.assertEqual(len(self.game.rules.deck), 52 - (num_players*2) - 1 - 3 - 1 - 1) # +1 burn, +1 turn

        self.game.play_stage('river')
        self.assertEqual(len(self.game.rules.community_cards), 5)
        self.assertIsInstance(self.game.rules.community_cards[4], str)
        self.assertEqual(len(self.game.rules.deck), 52 - (num_players*2) - 1 - 3 - 1 - 1 - 1 - 1) # +1 burn, +1 river

    def test_reset_hand(self):
        self.game.initialize_game()
        # In 2-player, P1 (SB) acts first.
        self.game.process_action(player_index=1, action="bet", raise_amount=100)
        self.game.play_stage('flop')

        dealer_before = self.game.rules.dealer_button
        chips_before_reinit = list(self.game.rules.player_chips)

        self.game.initialize_game()

        self.assertEqual(self.game.rules.dealer_button, dealer_before, "Dealer button should NOT change with just initialize_game.")
        self.assertEqual(len(self.game.rules.hands), self.game.rules.num_players)
        self.assertEqual(len(self.game.rules.community_cards), 0) # Fixed by change in initialize_game
        self.assertTrue(self.game.rules.pot > 0)

        sb_player_idx = (self.game.rules.dealer_button + 1) % self.game.rules.num_players
        bb_player_idx = (self.game.rules.dealer_button + 2) % self.game.rules.num_players
        expected_bets = [0] * self.game.rules.num_players
        expected_bets[sb_player_idx] = self.game_config['small_blind']
        expected_bets[bb_player_idx] = self.game_config['big_blind']
        self.assertEqual(list(self.game.rules.bets), expected_bets)

        expected_chips_after_reinit_blinds = list(chips_before_reinit)
        expected_chips_after_reinit_blinds[sb_player_idx] -= self.game_config['small_blind']
        expected_chips_after_reinit_blinds[bb_player_idx] -= self.game_config['big_blind']
        self.assertEqual(list(self.game.rules.player_chips), expected_chips_after_reinit_blinds)
        self.assertEqual(len(self.game.rules.deck), 52 - (self.game_config['num_players']*2))

    def test_reset_for_next_game_session(self):
        self.game.initialize_game()
        self.game.process_action(player_index=1, action="bet", raise_amount=100)
        self.game.play_stage('flop')

        dealer_before = self.game.rules.dealer_button
        self.assertTrue(hasattr(self.game, 'reset_for_next_hand'), "TexasHoldem should have reset_for_next_hand method")
        self.game.reset_for_next_hand()

        self.assertNotEqual(self.game.rules.dealer_button, dealer_before, "Dealer button should rotate after reset_for_next_hand")
        self.assertEqual(list(self.game.rules.player_chips),
                         [self.game_config['starting_stack']] * self.game_config['num_players'],
                         "Player chips should be reset to starting_stack by reset_for_next_hand")

        self.game.initialize_game()

        self.assertTrue(self.game.rules.pot > 0)
        expected_chips_after_blinds = [self.game_config['starting_stack']] * self.game_config['num_players']
        sb_player_idx = (self.game.rules.dealer_button + 1) % self.game.rules.num_players
        bb_player_idx = (self.game.rules.dealer_button + 2) % self.game.rules.num_players
        expected_chips_after_blinds[sb_player_idx] -= self.game_config['small_blind']
        expected_chips_after_blinds[bb_player_idx] -= self.game_config['big_blind']
        self.assertEqual(list(self.game.rules.player_chips), expected_chips_after_blinds)
        self.assertEqual(len(self.game.rules.deck), 52 - (self.game_config['num_players']*2))

    def test_betting_simple_fold(self):
        self.game.initialize_game()
        self.assertEqual(self.game.rules.current_player, 1)
        self.game.process_action(player_index=1, action="fold", raise_amount=None)
        self.assertTrue(self.game.end_game_early)
        self.assertEqual(self.game.winner, 0)
        self.game.declare_winner()
        self.assertEqual(self.game.rules.player_chips[1], self.game_config['starting_stack'] - self.game_config['small_blind'])
        self.assertEqual(self.game.rules.player_chips[0], self.game_config['starting_stack'] - self.game_config['big_blind'] + (self.game_config['small_blind'] + self.game_config['big_blind']))

    def test_betting_check_call_bet(self):
        self.game.initialize_game()
        self.assertEqual(self.game.rules.current_player, 1)
        self.assertEqual(self.game.rules.bets[0], self.game_config['big_blind'])
        self.assertEqual(self.game.rules.bets[1], self.game_config['small_blind'])
        self.assertEqual(self.game.rules.pot, self.game_config['small_blind'] + self.game_config['big_blind'])

        # P1 (SB) calls BB
        self.game.process_action(player_index=1, action="call", raise_amount=None)
        self.game.rules.actions_this_round += 1
        self.assertEqual(self.game.rules.player_chips[1], 1000 - self.game_config['big_blind'])
        self.assertEqual(self.game.rules.bets[1], self.game_config['big_blind'])
        expected_pot_after_p1_call = self.game_config['big_blind'] * 2
        self.assertEqual(self.game.rules.pot, expected_pot_after_p1_call)
        self.game.rules.advance_turn()
        self.assertEqual(self.game.rules.current_player, 0)

        # P0 (BB) checks
        self.game.process_action(player_index=0, action="check", raise_amount=None)
        self.game.rules.actions_this_round += 1
        self.assertEqual(self.game.rules.player_chips[0], 1000 - self.game_config['big_blind'])
        self.assertEqual(self.game.rules.bets[0], self.game_config['big_blind'])
        self.assertEqual(self.game.rules.pot, expected_pot_after_p1_call)
        self.assertTrue(self.game.rules.betting_round_is_over(), "Pre-flop betting round should be over")

        self.game.play_stage("flop")
        self.assertEqual(self.game.rules.current_player, 1)

        bet_amount_flop = 20
        self.game.process_action(player_index=1, action="bet", raise_amount=bet_amount_flop)
        self.game.rules.actions_this_round += 1
        self.assertEqual(self.game.rules.player_chips[1], 1000 - self.game_config['big_blind'] - bet_amount_flop)
        self.assertEqual(self.game.rules.bets[1], bet_amount_flop)
        self.assertEqual(self.game.rules.pot, expected_pot_after_p1_call + bet_amount_flop)
        self.assertEqual(self.game.rules.current_bet, bet_amount_flop)
        self.game.rules.advance_turn()
        self.assertEqual(self.game.rules.current_player, 0)

        self.game.process_action(player_index=0, action="call", raise_amount=None)
        self.game.rules.actions_this_round += 1
        self.assertEqual(self.game.rules.player_chips[0], 1000 - self.game_config['big_blind'] - bet_amount_flop)
        self.assertEqual(self.game.rules.bets[0], bet_amount_flop)
        self.assertEqual(self.game.rules.pot, expected_pot_after_p1_call + bet_amount_flop + bet_amount_flop)
        self.assertTrue(self.game.rules.betting_round_is_over(), "Flop betting round should be over")

if __name__ == '__main__':
    unittest.main()
