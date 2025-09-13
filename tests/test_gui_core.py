#!/usr/bin/env python3
"""
Test script for Texas Hold'em GUI - focuses on critical functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = pytest.mark.gui


class TestGUICore(unittest.TestCase):
    """Test core GUI functionality without running mainloop."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock tkinter to avoid creating actual windows
        self.tk_mock = Mock()
        self.root_mock = Mock()
        self.tk_mock.Tk.return_value = self.root_mock

    @patch("play.gui.tk")
    def test_gui_import_and_init(self, mock_tk):
        """Test that GUI can be imported and initialized."""
        mock_tk.Tk.return_value = self.root_mock

        from play.gui import GUIHumanStrategy

        # Test GUIHumanStrategy
        gui_instance = Mock()
        strategy = GUIHumanStrategy(gui_instance)
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.gui_instance, gui_instance)

    def test_card_image_key_conversion(self):
        """Test card image key conversion logic."""
        from play.gui import PokerGameGUI

        # Mock tkinter and mainloop to prevent actual GUI creation
        with (
            patch("tkinter.Tk") as mock_tk,
            patch.object(PokerGameGUI, "setup_gui"),
            patch.object(PokerGameGUI, "start_game"),
            patch.object(PokerGameGUI, "_load_card_images"),
        ):

            mock_root = Mock()
            mock_tk.return_value = mock_root
            mock_root.mainloop = Mock()  # Prevent actual mainloop

            gui = PokerGameGUI()

            # Test valid cards
            self.assertEqual(gui._get_card_image_key("As"), "As")
            self.assertEqual(gui._get_card_image_key("Kh"), "Kh")
            self.assertEqual(gui._get_card_image_key("2♠"), "2s")
            self.assertEqual(gui._get_card_image_key("J♥"), "Jh")

            # Test invalid cards
            self.assertIsNone(gui._get_card_image_key(""))
            self.assertIsNone(gui._get_card_image_key(None))
            self.assertIsNone(gui._get_card_image_key("X"))

            # Test placeholder cards
            self.assertEqual(gui._get_card_image_key("New Card"), "placeholder")
            self.assertEqual(gui._get_card_image_key("Card"), "placeholder")

    def test_game_logic_imports(self):
        """Test that all required game logic imports work."""
        from game_engine.texas_holdem import TexasHoldem

        from play.strategies import PlaceholderAIStrategy
        from playStrategy import HumanStrategy, RandomAIStrategy

        # Test creating strategies
        human_strategy = HumanStrategy()
        ai_strategy = RandomAIStrategy()
        placeholder_strategy = PlaceholderAIStrategy()

        self.assertIsNotNone(human_strategy)
        self.assertIsNotNone(ai_strategy)
        self.assertIsNotNone(placeholder_strategy)

        # Test creating a game
        strategies = [ai_strategy, ai_strategy]
        game = TexasHoldem(2, 1000, strategies)
        self.assertEqual(game.num_players, 2)


if __name__ == "__main__":
    print("Running GUI Core Tests...")
    unittest.main(verbosity=2)
