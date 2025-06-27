#!/usr/bin/env python3
"""
Test script to verify GUI integration with game logic
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from play.gui import PokerGameGUI
from playStrategy import HumanStrategy, RandomAIStrategy
from game_engine.texas_holdem import TexasHoldem

def test_gui_initialization():
    """Test that GUI can be initialized without errors"""
    try:
        # Create GUI without showing (for testing)
        gui = PokerGameGUI(None)
        print("✓ GUI initialization successful")
        return True
    except Exception as e:
        print(f"✗ GUI initialization failed: {e}")
        return False

def test_strategy_imports():
    """Test that strategy imports work correctly"""
    try:
        human_strategy = HumanStrategy()
        ai_strategy = RandomAIStrategy()
        print("✓ Strategy imports successful")
        print(f"  - HumanStrategy: {type(human_strategy).__name__}")
        print(f"  - RandomAIStrategy: {type(ai_strategy).__name__}")
        return True
    except Exception as e:
        print(f"✗ Strategy imports failed: {e}")
        return False

def test_game_engine_creation():
    """Test that game engine can be created with strategies"""
    try:
        # Test the same configuration that GUI would use
        total_players = 3  # 1 human + 2 AI
        starting_stack = 1000
        
        # Create strategies like the GUI does
        player_strategies = [HumanStrategy()]  # Human player
        for i in range(2):  # 2 AI players
            player_strategies.append(RandomAIStrategy())
          # Create game engine
        engine = TexasHoldem(
            num_players=total_players,
            starting_stack=starting_stack,
            player_strategies=player_strategies
        )
        
        print("✓ Game engine creation successful")
        print(f"  - Players: {total_players}")
        print(f"  - Starting stack: {starting_stack}")
        print(f"  - Strategies: {[type(s).__name__ for s in player_strategies]}")
        return True
    except Exception as e:
        print(f"✗ Game engine creation failed: {e}")
        return False

def test_gui_game_setup():
    """Test GUI's game setup method"""
    try:
        gui = PokerGameGUI(None)
        
        # Test the start_game_with_players method
        ai_count = 2
        starting_stack = 1500
        
        # This should create and initialize the game without errors
        gui.start_game_with_players(ai_count, starting_stack)
        
        print("✓ GUI game setup successful")
        print(f"  - AI count: {ai_count}")
        print(f"  - Starting stack: {starting_stack}")
        return True
    except Exception as e:
        print(f"✗ GUI game setup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing GUI Integration")
    print("=" * 50)
    
    tests = [
        test_strategy_imports,
        test_gui_initialization,
        test_game_engine_creation,
        test_gui_game_setup
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
