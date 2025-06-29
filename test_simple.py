#!/usr/bin/env python3
"""
Simple test to verify GUI imports and basic functionality
"""

import os
import sys

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

def test_imports():
    """Test all necessary imports"""
    try:
        from playStrategy import HumanStrategy, RandomAIStrategy
        print("✓ Strategy imports successful")
        
        from game_engine.texas_holdem import TexasHoldem
        print("✓ Game engine import successful")
        
        from play.gui import PokerGameGUI
        print("✓ GUI import successful")

        assert True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        assert False

def test_strategy_creation():
    """Test strategy object creation"""
    try:
        from playStrategy import HumanStrategy, RandomAIStrategy
        
        human = HumanStrategy()
        ai = RandomAIStrategy()
        
        print(f"✓ Created {type(human).__name__} and {type(ai).__name__}")
        assert True
    except Exception as e:
        print(f"✗ Strategy creation failed: {e}")
        assert False

def test_game_creation():
    """Test creating a game with strategies"""
    try:
        from playStrategy import HumanStrategy, RandomAIStrategy
        from game_engine.texas_holdem import TexasHoldem
        
        # Create strategies (1 human + 2 AI)
        strategies = [HumanStrategy(), RandomAIStrategy(), RandomAIStrategy()]
        
        # Create game
        game = TexasHoldem(
            num_players=3,
            starting_stack=1000,
            player_strategies=strategies
        )
        
        print("✓ Game creation successful")
        print(f"  - Players: {game.rules.num_players}")
        print(f"  - Starting stack: {game.rules.starting_stack}")
        assert True
    except Exception as e:
        print(f"✗ Game creation failed: {e}")
        assert False

def main():
    """Run tests"""
    print("Testing GUI Integration Components")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_strategy_creation,
        test_game_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All core components working!")
        print("\nThe GUI should be ready to run. Try:")
        print("  python play/gui.py")
    else:
        print("❌ Some tests failed.")

if __name__ == "__main__":
    main()
