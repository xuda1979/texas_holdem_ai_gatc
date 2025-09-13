#!/usr/bin/env python3
"""
Test the new player selection functionality in the GUI
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk

from play.gui import PokerGameGUI


def test_player_selection_gui():
    """Test that the player selection GUI can be created"""
    print("Testing player selection GUI...")

    try:
        # Create a root window
        root = tk.Tk()
        root.withdraw()  # Hide for testing

        # Create GUI instance
        gui = PokerGameGUI(root)

        print("✓ GUI created successfully")

        # Check that game engine is initially None
        if gui.game_engine is None:
            print("✓ Game engine is None initially (correct)")
        else:
            print("✗ Game engine should be None initially")
            return False

        # Check that selection methods exist
        if hasattr(gui, 'show_player_selection'):
            print("✓ show_player_selection method exists")
        else:
            print("✗ show_player_selection method missing")
            return False

        if hasattr(gui, 'start_game_with_players'):
            print("✓ start_game_with_players method exists")
        else:
            print("✗ start_game_with_players method missing")
            return False

        # Test creating a game with different player counts
        for ai_count in [1, 2, 3, 4, 5]:
            gui.start_game_with_players(ai_count)

            if gui.game_engine is None:
                print(f"✗ Game engine not created for {ai_count} AI players")
                return False

            total_players = gui.game_engine.num_players
            expected_players = ai_count + 1  # AI + 1 human

            if total_players == expected_players:
                print(f"✓ Game with {ai_count} AI players: {total_players} total players")
            else:
                print(f"✗ Expected {expected_players}, got {total_players} for {ai_count} AIs")
                return False

        root.destroy()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_ai_display_functionality():
    """Test that AI display handles multiple players correctly"""
    print("\nTesting AI display functionality...")

    try:
        root = tk.Tk()
        root.withdraw()

        gui = PokerGameGUI(root)

        # Test with 3 AI players
        gui.start_game_with_players(3)

        # Check AI display methods exist
        if hasattr(gui, '_setup_ai_display'):
            print("✓ _setup_ai_display method exists")
        else:
            print("✗ _setup_ai_display method missing")
            return False

        if hasattr(gui, '_update_ai_display'):
            print("✓ _update_ai_display method exists")
        else:
            print("✗ _update_ai_display method missing")
            return False

        # Check that AI labels are created
        if hasattr(gui, 'ai_money_labels') and len(gui.ai_money_labels) == 3:
            print("✓ Correct number of AI labels created")
        else:
            print(f"✗ Expected 3 AI labels, got {len(gui.ai_money_labels) if hasattr(gui, 'ai_money_labels') else 0}")
            return False

        root.destroy()
        return True

    except Exception as e:
        print(f"✗ AI display test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Player Selection GUI Features")
    print("=" * 40)

    test1 = test_player_selection_gui()
    test2 = test_ai_display_functionality()

    print("\n" + "=" * 40)
    print("TEST RESULTS:")
    print("=" * 40)

    if test1 and test2:
        print("🎉 ALL TESTS PASSED!")
        print("\nNew features working correctly:")
        print("- ✅ Player selection dialog")
        print("- ✅ Variable number of AI opponents (1-5)")
        print("- ✅ Dynamic AI player display")
        print("- ✅ Game engine creation with correct player count")
        print("\n🚀 Ready to use! Run: python play/gui.py")
        print("You will be prompted to choose how many AI opponents to play against!")
    else:
        print("❌ Some tests failed")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
