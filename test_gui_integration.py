#!/usr/bin/env python3
"""
Test script to verify GUI integration with game logic
"""

import os
import sys
from unittest.mock import patch

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from game_engine.texas_holdem import TexasHoldem  # noqa: E402
from play.gui import PokerGameGUI  # noqa: E402
from playStrategy import HumanStrategy, RandomAIStrategy  # noqa: E402


def test_gui_initialization() -> None:
    """Test that GUI can be initialized without errors"""
    try:
        with (
            patch("play.gui.tk") as mock_tk,
            patch("play.gui.PokerGameGUI._load_card_images"),
            patch("play.gui.PokerGameGUI.setup_gui"),
            patch("play.gui.PokerGameGUI.start_game"),
        ):
            _gui = PokerGameGUI()
            mock_tk.Tk.assert_called_once()
        print("✓ GUI initialization successful")
        assert True
    except Exception as e:
        print(f"✗ GUI initialization failed: {e}")
        raise AssertionError() from e


def test_strategy_imports() -> None:
    """Test that strategy imports work correctly"""
    try:
        human_strategy = HumanStrategy()
        ai_strategy = RandomAIStrategy()
        print("✓ Strategy imports successful")
        print(f"  - HumanStrategy: {type(human_strategy).__name__}")
        print(f"  - RandomAIStrategy: {type(ai_strategy).__name__}")
        assert True
    except Exception as e:
        print(f"✗ Strategy imports failed: {e}")
        raise AssertionError() from e


def test_game_engine_creation() -> None:
    """Test that game engine can be created with strategies"""
    try:
        # Test the same configuration that GUI would use
        total_players = 3  # 1 human + 2 AI
        starting_stack = 1000

        # Create strategies like the GUI does
        player_strategies = [HumanStrategy()]  # Human player
        for _ in range(2):  # 2 AI players
            player_strategies.append(RandomAIStrategy())

        # Create game engine
        _engine = TexasHoldem(
            num_players=total_players,
            starting_stack=starting_stack,
            player_strategies=player_strategies,
        )

        print("✓ Game engine creation successful")
        print(f"  - Players: {total_players}")
        print(f"  - Starting stack: {starting_stack}")
        print(f"  - Strategies: {[type(s).__name__ for s in player_strategies]}")
        assert True
    except Exception as e:
        print(f"✗ Game engine creation failed: {e}")
        raise AssertionError() from e


def test_gui_game_setup() -> None:
    """Test GUI's game setup method"""
    try:
        with (
            patch("play.gui.tk") as mock_tk,
            patch("play.gui.PokerGameGUI._load_card_images"),
            patch("play.gui.PokerGameGUI.setup_gui"),
            patch("play.gui.PokerGameGUI.start_game"),
        ):
            gui = PokerGameGUI()

            ai_count = 2
            starting_stack = 1500

            gui.start_game_with_players(ai_count, starting_stack)
            mock_tk.Tk.assert_called()

        print("✓ GUI game setup successful")
        print(f"  - AI count: {ai_count}")
        print(f"  - Starting stack: {starting_stack}")
        assert True
    except Exception as e:
        print(f"✗ GUI game setup failed: {e}")
        raise AssertionError() from e


def main() -> None:
    """Run all tests"""
    print("Testing GUI Integration")
    print("=" * 50)

    tests = [
        test_strategy_imports,
        test_gui_initialization,
        test_game_engine_creation,
        test_gui_game_setup,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! GUI integration is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
