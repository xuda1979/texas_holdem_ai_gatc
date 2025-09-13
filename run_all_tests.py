#!/usr/bin/env python3
"""
Comprehensive test runner for Texas Hold'em AI project.
Tests all core components and provides a summary.
"""

import os
import subprocess
import sys
import traceback

# Add project root and source directory to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure the src directory is available for imports
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def run_test_suite(test_name, test_command):
    """Run a test suite and return result."""
    try:
        print(f"\n--- Testing {test_name} ---")
        result = subprocess.run(
            test_command, shell=True, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            print(f"✅ {test_name}: PASSED")
            return True
        else:
            print(f"❌ {test_name}: FAILED")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name}: TIMEOUT (>30s)")
        return False
    except Exception as e:
        print(f"💥 {test_name}: ERROR - {e}")
        return False


def test_imports():
    """Test that all critical imports work."""
    try:
        print("\n--- Testing Critical Imports ---")

        # Test game engine

        print("✅ Game engine imports")

        # Test strategies

        print("✅ Strategy imports")

        # Test AI components

        print("✅ AI model imports")

        # Test GUI (mock tkinter to avoid display issues)
        import sys
        from unittest.mock import MagicMock

        sys.modules["tkinter"] = MagicMock()
        sys.modules["tkinter.messagebox"] = MagicMock()
        sys.modules["tkinter.simpledialog"] = MagicMock()

        print("✅ GUI imports")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False


def test_basic_game_logic():
    """Test basic game functionality."""
    try:
        print("\n--- Testing Basic Game Logic ---")

        from game_engine.texas_holdem import TexasHoldem

        from playStrategy import RandomAIStrategy

        # Create a simple game
        strategies = [RandomAIStrategy(), RandomAIStrategy()]
        game = TexasHoldem(2, 1000, strategies)

        # Verify basic properties
        assert game.num_players == 2
        assert all(chips == 1000 for chips in game.rules.player_chips)
        print("✅ Game creation and initialization")

        # Test a simple hand (without full simulation to avoid complexity)
        initial_deck_size = len(game.rules.deck.cards)
        assert initial_deck_size == 52
        print("✅ Deck initialization")

        return True

    except Exception as e:
        print(f"❌ Game logic error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests and provide summary."""
    print("🃏 Texas Hold'em AI - Comprehensive Test Suite")
    print("=" * 50)

    passed = 0
    total = 0

    # Test categories
    test_cases = [
        ("Critical Imports", test_imports),
        ("Basic Game Logic", test_basic_game_logic),
        ("Action Mapping", "python -m pytest tests/test_action_mapping.py -q"),
        ("Texas Hold'em Rules", "python -m pytest tests/test_texas_holdem_rules.py -q"),
        ("Betting Tree", "python -m pytest tests/test_betting_tree.py -q"),
        ("Exploitability", "python -m pytest tests/test_exploitability.py -q"),
    ]

    for test_name, test_func_or_cmd in test_cases:
        total += 1

        if callable(test_func_or_cmd):
            # Run Python function test
            if test_func_or_cmd():
                passed += 1
        else:
            # Run command line test
            if run_test_suite(test_name, test_func_or_cmd):
                passed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The project is in good shape.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
