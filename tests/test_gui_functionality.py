# ruff: noqa
import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

import pytest

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = pytest.mark.gui

# Mock tkinter before importing GUI
try:
    import tkinter  # noqa: F401
except Exception:
    sys.modules["tkinter"] = MagicMock()
    sys.modules["tkinter.messagebox"] = MagicMock()
    sys.modules["tkinter.simpledialog"] = MagicMock()

from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.gui import GUIHumanStrategy, PokerGameGUI
from poker_ai.gui.playStrategy import RandomAIStrategy


class TestPokerGameGUIFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up a test environment before each test."""
        # Patch the entire tkinter module
        self.patcher_tk = patch("poker_ai.gui.gui.tk")
        self.mock_tk = self.patcher_tk.start()

        # Patch Pillow
        self.patcher_pil = patch("poker_ai.gui.gui.Image", autospec=True)
        self.mock_image = self.patcher_pil.start()

        # Prevent the main loop from running
        self.patcher_mainloop = patch.object(self.mock_tk.Tk.return_value, "mainloop")
        self.mock_mainloop = self.patcher_mainloop.start()

        # Patch image loading method
        self.patcher_load_images = patch("poker_ai.gui.gui.PokerGameGUI._load_card_images")
        self.mock_load_images = self.patcher_load_images.start()

        # Patch messagebox to avoid display errors
        self.patcher_messagebox = patch("poker_ai.gui.gui.messagebox")
        self.mock_messagebox = self.patcher_messagebox.start()

        # Create an instance of the GUI
        with patch.object(PokerGameGUI, "setup_initial_gui") as mock_setup:
            self.gui = PokerGameGUI()
        mock_setup.assert_called_once()

    def tearDown(self):
        """Clean up after each test."""
        self.patcher_tk.stop()
        self.patcher_pil.stop()
        self.patcher_mainloop.stop()
        self.patcher_load_images.stop()
        self.patcher_messagebox.stop()

    def test_01_gui_initialization(self):
        """Test that the GUI initializes correctly."""
        print("\nRunning test_01_gui_initialization...")
        self.mock_tk.Tk.assert_called_once()
        self.gui.root.title.assert_called_with("Texas Hold'em Poker - Human vs AI")
        self.gui.root.geometry.assert_called_with("1400x900")
        self.assertTrue(self.mock_load_images.called)
        print("✓ GUI initialization successful.")

    def test_02_start_new_game(self):
        """Test the start_new_game functionality."""
        print("\nRunning test_02_start_new_game...")
        self.gui.total_players_var = MagicMock()
        self.gui.total_players_var.get.return_value = "3"
        self.gui.starting_stack_var = MagicMock()
        self.gui.starting_stack_var.get.return_value = "5000"

        with (
            patch("poker_ai.gui.gui.TexasHoldem") as MockTexasHoldem,
            patch.object(self.gui, "setup_game_gui") as mock_setup_game_gui,
            patch.object(self.gui, "play_hand") as mock_play_hand,
        ):

            self.gui.start_new_game()

            # Verify game creation
            MockTexasHoldem.assert_called_once()
            args, kwargs = MockTexasHoldem.call_args
            self.assertEqual(args[0], 3)
            self.assertEqual(args[1], 5000)
            self.assertEqual(len(args[2]), 3)
            self.assertIsInstance(args[2][0], GUIHumanStrategy)
            self.assertIsInstance(args[2][1], RandomAIStrategy)

            # Verify GUI and game loop start
            mock_setup_game_gui.assert_called_once()
            mock_play_hand.assert_called_once()
        print("✓ start_new_game successful.")

    def test_03_update_display_with_cards(self):
        """Test that update_display shows player and community cards."""
        print("\nRunning test_03_update_display_with_cards...")
        # Setup a mock game object
        self.gui.game = MagicMock()
        self.gui.game.num_players = 3
        self.gui.game.rules.player_chips = [1000, 1000, 1000]
        self.gui.game.rules.community_cards = ["As", "Kd", "Qc"]
        self.gui.game.rules.hands = [["Ah", "Kh"], ["Jd", "Js"], ["Ts", "9s"]]
        self.gui.game.rules.pot = 500
        self.gui.game.rules.bets = [50, 100, 100]
        self.gui.game.rules.current_bet = 100

        # Mock frames
        self.gui.info_frame = self.mock_tk.Frame()
        self.gui.cards_frame = self.mock_tk.Frame()
        self.gui.player_frame = self.mock_tk.Frame()

        with patch.object(self.gui, "create_card_label") as mock_create_card_label:
            self.gui.update_display()

            # Check community cards
            expected_community_calls = [
                call(unittest.mock.ANY, "As"),
                call(unittest.mock.ANY, "Kd"),
                call(unittest.mock.ANY, "Qc"),
            ]
            mock_create_card_label.assert_has_calls(expected_community_calls, any_order=True)

            # Check player hand
            expected_hand_calls = [call(unittest.mock.ANY, "Ah"), call(unittest.mock.ANY, "Kh")]
            mock_create_card_label.assert_has_calls(expected_hand_calls, any_order=True)

        print("✓ update_display with cards successful.")

    def test_04_human_action_raise(self):
        """Test the human_action method for a raise."""
        print("\nRunning test_04_human_action_raise...")
        self.gui.game = MagicMock(spec=TexasHoldem)
        self.gui.human_strategy = MagicMock(spec=GUIHumanStrategy)

        with patch("poker_ai.gui.gui.simpledialog.askinteger") as mock_askinteger:
            mock_askinteger.return_value = 200
            self.gui.game.get_min_raise_amount.return_value = 100
            self.gui.game.get_max_raise_amount.return_value = 1000

            self.gui.human_action("raise")

            mock_askinteger.assert_called_once()
            self.gui.human_strategy.set_action.assert_called_with("raise", 200)
        print("✓ human_action for raise successful.")

    def test_05_play_hand_and_next_hand(self):
        """Test the main hand playing and advancing logic."""
        print("\nRunning test_05_play_hand_and_next_hand...")
        self.gui.game = MagicMock(spec=TexasHoldem)
        self.gui.status_label = self.mock_tk.Label()
        self.gui.actions_frame = self.mock_tk.Frame()

        # Test play_hand
        self.gui.play_hand()
        self.gui.game.play_game.assert_called_once()
        self.assertTrue(self.mock_tk.Button.called)
        last_button_text = self.mock_tk.Button.call_args[1]["text"]
        self.assertEqual(last_button_text, "Next Hand")

        # Test next_hand
        with patch.object(self.gui, "play_hand") as mock_play_hand:
            self.gui.next_hand()
            self.gui.game.reset_for_next_hand.assert_called_once()
            mock_play_hand.assert_called_once()
        print("✓ play_hand and next_hand logic successful.")


if __name__ == "__main__":
    unittest.main()
