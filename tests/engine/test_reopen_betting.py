
import pytest
from unittest.mock import MagicMock
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.rules.texas_holdem import TexasHoldemRules

class TestBettingReopening:
    def test_all_in_full_raise_reopens_betting(self):
        """
        Test that an all-in raise that meets the minimum raise requirement reopens the betting.
        """
        game = TexasHoldem(num_players=3, starting_stack=1000, verbose=True)
        # Setup specific chip stacks
        game.rules.player_chips = [1000, 340, 1000] # P1 (SB), P2 (BB), P3 (UTG)

        # Pre-flop
        # P1 posts 10
        # P2 posts 20
        # P3 (UTG) raises to 40
        game.process_action(2, "raise", 40)
        # P1 calls 40
        game.process_action(0, "call")
        # P2 calls 40. Chips left: 340 - 40 = 300.
        game.process_action(1, "call")

        # Advance to Flop
        game.play_stage("flop")

        # Flop betting
        # P1 (SB) checks
        game.process_action(0, "check")

        # P2 (BB) checks
        game.process_action(1, "check")

        # P3 (UTG) bets 100
        game.process_action(2, "bet", 100)

        # P1 (SB) calls 100
        game.process_action(0, "call")

        # P2 (BB) goes ALL-IN for 300 total (Raise to 300).
        # Original bet was 100. Raise to 300 is a raise of 200.
        # Min raise was 100 (bet 100).
        # So 200 >= 100. This is a full raise.
        # It should reopen betting for P3.

        assert game.rules.player_chips[1] == 300

        game.process_action(1, "raise", 300)

        # Now check if P3 is allowed to raise again.
        valid_actions = game.get_valid_actions(2)
        assert "raise" in valid_actions, f"Betting should be reopened for Player 3, but valid actions are {valid_actions}"

    def test_short_all_in_does_not_reopen_betting(self):
        """
        Test that a short all-in raise (less than min raise) does NOT reopen betting for players who already acted.
        """
        game = TexasHoldem(num_players=3, starting_stack=1000, verbose=True)
        game.rules.player_chips = [150, 1000, 1000]

        # Fast forward to flop (assume preflop checks/limps to keep stacks simple)
        # Hack: just set pot and deal cards manually-ish or just run through preflop
        game.rules.pot = 30
        game.rules.bets = [0,0,0]
        game.rules.active_players = [True, True, True]
        game.rules.dealer_button = 2 # P1 is SB, P2 BB, P3 Button

        # Manually set stage to flop
        game.rules.betting_round = "flop"
        game.rules.current_player = 0

        # P1 checks
        game.process_action(0, "check")

        # P2 bets 100
        game.process_action(1, "bet", 100)

        # P3 calls 100
        game.process_action(2, "call")

        # Back to P1. P1 has 150 chips.
        # P1 raises All-in to 150.
        # Raise amount is 50. Min raise is 100.
        # So this is incomplete.
        game.process_action(0, "raise", 150)

        # Action is on P2.
        # P2 initiated the bet of 100.
        # He is facing a raise of 50.
        # Since it's incomplete, he cannot re-raise.
        valid_actions_p2 = game.get_valid_actions(1)

        # "raise" should NOT be in valid actions for P2
        assert "raise" not in valid_actions_p2, f"Betting should NOT be reopened for Player 2, but valid actions are {valid_actions_p2}"
        assert "call" in valid_actions_p2
        assert "fold" in valid_actions_p2
