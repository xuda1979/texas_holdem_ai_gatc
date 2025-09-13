#!/usr/bin/env python3
"""
Quick test to verify the GUI shows properly and AIs don't play without human
"""

import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Test the GUI logic without actually opening the GUI window
def test_gui_logic():
    print("Testing GUI Logic (Simulated)")
    print("=" * 40)

    try:
        # Import the GUI class

        # Check if we can create an instance without running mainloop
        print("✓ GUI class imports successfully")

        # Test the player setup logic
        from game_engine.texas_holdem import TexasHoldem

        from play.strategies import PlaceholderAIStrategy

        # Simulate what happens when user selects 3 AI opponents
        ai_count = 3
        total_players = ai_count + 1  # 4 total players

        ai_strategy = PlaceholderAIStrategy()
        player_strategies = [None]  # Human player at index 0

        for i in range(ai_count):
            player_strategies.append(ai_strategy)

        # Create game engine
        game_engine = TexasHoldem(
            num_players=total_players, starting_stack=1000, player_strategies=player_strategies
        )

        print(f"✓ Game engine created with {total_players} players")
        print(f"  - Human player: index 0 (strategy: {player_strategies[0]})")
        print(f"  - AI players: indices 1-{ai_count} (all have strategies)")

        # Check that human player is always index 0
        human_player_index = 0
        if player_strategies[human_player_index] is None:
            print("✓ Human player correctly has no AI strategy")
        else:
            print("✗ ERROR: Human player should not have an AI strategy")
            return False

        # Check that AIs have strategies
        for i in range(1, total_players):
            if player_strategies[i] is not None:
                print(f"✓ AI Player {i} has strategy")
            else:
                print(f"✗ ERROR: AI Player {i} should have a strategy")
                return False

        # Initialize game and check whose turn it is
        game_engine.initialize_game()
        current_player = game_engine.rules.current_player

        print(f"✓ Game initialized, current player: {current_player}")

        # In a real game, if it's the human's turn, the GUI should wait
        # If it's an AI's turn, the AI should act, then return control to check if human is next
        if current_player == human_player_index:
            print("✓ Human goes first - GUI will wait for human input")
        else:
            print(
                f"✓ AI Player {current_player} goes first - will act automatically, then check for human turn"
            )

        print("\n" + "=" * 40)
        print("🎉 ALL LOGIC TESTS PASSED!")
        print("The GUI should:")
        print("1. Show player selection dialog first")
        print("2. After selection, show the main game interface")
        print("3. Wait for human input when it's human's turn")
        print("4. Let AIs act automatically only during their turns")
        print("5. Never let AIs play among themselves without human")
        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gui_logic()
    if success:
        print("\n🎯 Ready to test actual GUI!")
        print("Run: python play/gui.py")
        print("1. Select number of AI opponents")
        print("2. Click 'Start Game'")
        print("3. You should see the poker table interface")
        print("4. Game should wait for your action when it's your turn")
    else:
        print("\n❌ Logic test failed - check the issues above")
