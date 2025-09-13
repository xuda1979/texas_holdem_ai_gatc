import os
import sys
import unittest
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

# Conditional import of PokerGameGUI for testing
# This allows the test file to be parsed even if tkinter is not available in the environment
# The actual tests will mock out tkinter.
try:
    import tkinter  # noqa: F401
except Exception:
    sys.modules['tkinter'] = MagicMock()
    sys.modules['tkinter.font'] = MagicMock()
    sys.modules['tkinter.messagebox'] = MagicMock()

from poker_ai.gui.gui import PokerGameGUI

# Mock game_engine.texas_holdem before PokerGameGUI tries to import it at class level or __init__
# This is a common pattern if the import itself needs to be mocked early.
# However, PokerGameGUI imports it at module level.
# So, we need to ensure that when poker_ai.gui.gui is imported, it finds a mocked TexasHoldem.
# The best place for this is often at the top of the test file or in a setUpModule.

# For simplicity in this environment, we will patch it where it's looked up by poker_ai.gui.gui
# This can be done per test method or per test class using decorators.

# Placeholder for game_engine.texas_holdem.TexasHoldemRules if needed for type hints
# or direct instantiation in tests, though it's usually better to mock instances.
MockTexasHoldemRules = MagicMock()

class TestPokerGameGUI(unittest.TestCase):

    @patch('poker_ai.gui.gui.PokerGameGUI.start_game') # Patch start_game to prevent its execution
    @patch('poker_ai.gui.gui.PokerGameGUI.setup_gui') # Patch setup_gui to prevent its execution
    @patch('poker_ai.gui.gui.TexasHoldem', autospec=True)
    @patch('poker_ai.gui.gui.PokerGameGUI._load_card_images')
    @patch('poker_ai.gui.gui.tk')
    def test_gui_initialization(self, mock_tk, mock_load_images, MockTexasHoldem, mock_setup_gui, mock_start_game):
        """Test PokerGameGUI __init__ method."""
        print("\nRunning test_gui_initialization...")

        mock_engine_instance = MockTexasHoldem.return_value
        mock_engine_instance.rules = MagicMock()
        mock_engine_instance.rules.player_chips = [1000, 1000]
        mock_engine_instance.num_players = 2

        mock_root_instance = mock_tk.Tk.return_value # Get the mock root from mock_tk

        gui = PokerGameGUI()

        self.assertIsNotNone(gui.root)
        mock_tk.Tk.assert_called_once()

        # Game engine should be initialized lazily; start_game is patched
        MockTexasHoldem.assert_not_called()

        self.assertEqual(gui.human_player_index, 0)
        self.assertIsNotNone(gui.game_engine)
        mock_load_images.assert_called_once()
        mock_setup_gui.assert_called_once() # Verify setup_gui was called
        mock_start_game.assert_called_once() # Verify start_game was called

        mock_root_instance.mainloop.assert_called_once()

        if hasattr(gui.root, 'destroy'):
            gui.root.destroy()
        print("test_gui_initialization finished.")


    @patch('poker_ai.gui.gui.TexasHoldem', autospec=True)
    @patch('poker_ai.gui.gui.PokerGameGUI._load_card_images')
    @patch('poker_ai.gui.gui.tk')
    def test_get_card_image_key(self, mock_tk, mock_load_images, MockTexasHoldem):
        """Test the _get_card_image_key method."""
        print("\nRunning test_get_card_image_key...")

        # Prevent __init__ from running real setup_gui/start_game for this unit test
        with patch.object(PokerGameGUI, 'setup_gui', MagicMock()), \
             patch.object(PokerGameGUI, 'start_game', MagicMock()):

            mock_engine_instance = MockTexasHoldem.return_value
            mock_engine_instance.rules = MagicMock()
            mock_engine_instance.rules.player_chips = [1000,1000]
            mock_engine_instance.num_players = 2
            # Crucially define attributes expected by start_game if it were real, or by other init parts
            mock_engine_instance.end_game_early = False # Define this attribute

            gui = PokerGameGUI()

        self.assertEqual(gui._get_card_image_key("A♠"), "As")
        self.assertEqual(gui._get_card_image_key("K♦"), "Kd")
        self.assertEqual(gui._get_card_image_key("Q♥"), "Qh")
        self.assertEqual(gui._get_card_image_key("J♣"), "Jc")
        self.assertEqual(gui._get_card_image_key("T♠"), "Ts") # Assuming 'T' for Ten
        self.assertEqual(gui._get_card_image_key("9d"), "9d")
        self.assertEqual(gui._get_card_image_key("2c"), "2c")

        self.assertEqual(gui._get_card_image_key("New Card"), "placeholder")
        self.assertEqual(gui._get_card_image_key("Card"), "placeholder")

        self.assertIsNone(gui._get_card_image_key("Xy")) # Invalid format
        self.assertIsNone(gui._get_card_image_key("A")) # Too short
        self.assertIsNone(gui._get_card_image_key("A_")) # Invalid suit
        self.assertIsNone(gui._get_card_image_key(None))
        self.assertIsNone(gui._get_card_image_key(""))

        if hasattr(gui.root, 'destroy'): gui.root.destroy()
        print("test_get_card_image_key finished.")


    @patch('poker_ai.gui.gui.TexasHoldem', autospec=True)
    @patch('poker_ai.gui.gui.PokerGameGUI._load_card_images')
    @patch('poker_ai.gui.gui.tk')
    @patch('poker_ai.gui.gui.messagebox')
    def test_handle_player_actions(self, mock_messagebox, mock_tk, mock_load_images, MockTexasHoldem):
        """Test _handle_player_action for various actions."""
        print("\nRunning test_handle_player_actions...")

        # To prevent __init__ from running real setup_gui/start_game
        with patch.object(PokerGameGUI, 'setup_gui', MagicMock()), \
             patch.object(PokerGameGUI, 'start_game', MagicMock()):

            mock_engine_instance = MockTexasHoldem.return_value
            mock_engine_instance.rules = MagicMock()
            mock_engine_instance.rules.player_chips = [1000, 1000]
            mock_engine_instance.num_players = 2

            gui = PokerGameGUI()
            gui.game_engine = mock_engine_instance

            gui._sync_gui_with_engine_state = MagicMock()
            gui._handle_game_progression = MagicMock()

            # --- Test 'bet' action ---
            gui.game_engine.rules.current_player = 0 # Human's turn
            gui._handle_player_action('bet', 100)
            gui.game_engine.process_action.assert_called_with(0, 'bet', raise_amount=100)
            gui._sync_gui_with_engine_state.assert_called_with()
            gui._handle_game_progression.assert_called_with()
            gui.game_engine.process_action.reset_mock()
            gui._sync_gui_with_engine_state.reset_mock()
            gui._handle_game_progression.reset_mock()

            # --- Test 'call' action ---
            gui.game_engine.rules.current_player = 0 # Human's turn
            gui._handle_player_action('call')
            gui.game_engine.process_action.assert_called_with(0, 'call', raise_amount=None)
            gui._sync_gui_with_engine_state.assert_called_with()
            gui._handle_game_progression.assert_called_with()
            gui.game_engine.process_action.reset_mock()
            gui._sync_gui_with_engine_state.reset_mock()
            gui._handle_game_progression.reset_mock()

            # --- Test 'fold' action ---
            gui.game_engine.rules.current_player = 0 # Human's turn
            gui._handle_player_action('fold')
            gui.game_engine.process_action.assert_called_with(0, 'fold', raise_amount=None)
            gui._sync_gui_with_engine_state.assert_called_with()
            gui._handle_game_progression.assert_called_with()
            gui.game_engine.process_action.reset_mock()
            gui._sync_gui_with_engine_state.reset_mock()
            gui._handle_game_progression.reset_mock()

            # --- Test 'check' action ---
            gui.game_engine.rules.current_player = 0 # Human's turn
            gui._handle_player_action('check')
            gui.game_engine.process_action.assert_called_with(0, 'check', raise_amount=None)
            gui._sync_gui_with_engine_state.assert_called_with()
            gui._handle_game_progression.assert_called_with()
            gui.game_engine.process_action.reset_mock()
            gui._sync_gui_with_engine_state.reset_mock()
            gui._handle_game_progression.reset_mock()

            # --- Test action when not player's turn ---
            gui.game_engine.rules.current_player = 1 # AI's turn
            gui._handle_player_action('bet', 50) # Try to bet
            mock_messagebox.showwarning.assert_called_with("Not your turn", "It's not your turn to act.")
            gui.game_engine.process_action.assert_not_called() # Should not be called if not player's turn

            if hasattr(gui.root, 'destroy'): gui.root.destroy()
        print("test_handle_player_actions finished.")

    @patch('poker_ai.gui.gui.TexasHoldem', autospec=True)
    @patch('poker_ai.gui.gui.PokerGameGUI._load_card_images')
    @patch('poker_ai.gui.gui.tk')
    def test_sync_gui_with_engine_state(self, mock_tk, mock_load_images, MockTexasHoldem):
        """Test the _sync_gui_with_engine_state method."""
        print("\nRunning test_sync_gui_with_engine_state...")

        # Prevent __init__ from running real setup_gui/start_game for this unit test
        with patch.object(PokerGameGUI, 'setup_gui', MagicMock()), \
             patch.object(PokerGameGUI, 'start_game', MagicMock()):

            mock_engine_instance = MockTexasHoldem.return_value
            mock_engine_instance.rules = MagicMock() # This is the game_engine.rules mock
            mock_engine_instance.rules.player_chips = [1000, 800]
            mock_engine_instance.num_players = 2
            mock_engine_instance.rules.hands = [['Ah', 'Kh'], ['Ad', 'Kd']]
            mock_engine_instance.rules.community_cards = ['2s', '3s', '4s']
            mock_engine_instance.rules.pot = 500
            # Define attributes expected by other parts if start_game was real
            mock_engine_instance.end_game_early = False

            gui = PokerGameGUI()
            gui.game_engine = mock_engine_instance # Assign the fully configured mock engine

        gui.update_display = MagicMock() # Mock methods called by _sync_gui_with_engine_state
        gui._update_action_buttons_state = MagicMock()

        gui._sync_gui_with_engine_state()

        self.assertEqual(gui.player_hand, ['Ah', 'Kh'])
        self.assertEqual(gui.community_cards, ['2s', '3s', '4s'])
        self.assertEqual(gui.pot, 500)
        self.assertEqual(gui.player_money, 1000)
        self.assertEqual(gui.ai_money, 800)
        gui.update_display.assert_called_once()
        gui._update_action_buttons_state.assert_called_once()

        if hasattr(gui.root, 'destroy'): gui.root.destroy()
        print("test_sync_gui_with_engine_state finished.")


if __name__ == '__main__':
    unittest.main()
