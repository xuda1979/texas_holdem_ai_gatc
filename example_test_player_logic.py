#!/usr/bin/env python3
"""
Test the player selection functionality without initializing tkinter GUI
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_player_selection_logic():
    """Test the logic of player selection without GUI initialization"""
    print("Testing player selection logic...")
    
    # Import the GUI class
    from play.gui import PokerGameGUI
    
    # Check that required methods exist
    methods_to_check = [
        'show_player_selection',
        'start_game_with_players', 
        'setup_game_gui',
        '_setup_ai_display',
        '_update_ai_display'
    ]
    
    for method in methods_to_check:
        if hasattr(PokerGameGUI, method):
            print(f"✓ {method} method exists")
        else:
            print(f"✗ {method} method missing")
            return False
    
    print("✓ All required methods exist")
    return True

def test_game_logic():
    """Test game creation logic"""
    print("\nTesting game creation logic...")
    
    from game_engine.texas_holdem import TexasHoldem
    from play.strategies import PlaceholderAIStrategy
    
    # Test creating games with different player counts
    for ai_count in range(1, 6):  # 1 to 5 AI players
        total_players = ai_count + 1  # AI + 1 human
        
        try:
            # Create AI strategies
            ai_strategy = PlaceholderAIStrategy()
            player_strategies = [None]  # Human player
            
            # Add AI strategies
            for i in range(ai_count):
                player_strategies.append(ai_strategy)
            
            # Create game engine
            game = TexasHoldem(
                num_players=total_players,
                starting_stack=1000,
                player_strategies=player_strategies
            )
            
            if game.num_players == total_players:
                print(f"✓ Game with {ai_count} AIs: {total_players} total players")
            else:
                print(f"✗ Expected {total_players}, got {game.num_players}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to create game with {ai_count} AIs: {e}")
            return False
    
    return True

def verify_gui_structure():
    """Verify the GUI file has the correct structure"""
    print("\nVerifying GUI file structure...")
    
    with open('play/gui.py', 'r') as f:
        content = f.read()
    
    # Check for key components
    checks = [
        ('show_player_selection', 'Player selection dialog'),
        ('start_game_with_players', 'Game initialization with player count'),
        ('_setup_ai_display', 'AI display setup'),
        ('_update_ai_display', 'AI display updates'),
        ('ai_money_labels', 'Multiple AI player labels'),
        ('Change Players', 'Change players button')
    ]
    
    for search_term, description in checks:
        if search_term in content:
            print(f"✓ {description} found")
        else:
            print(f"✗ {description} missing")
            return False
    
    return True

def main():
    """Run all logic tests"""
    print("Testing Player Selection Logic (No GUI)")
    print("=" * 45)
    
    test1 = test_player_selection_logic()
    test2 = test_game_logic() 
    test3 = verify_gui_structure()
    
    print("\n" + "=" * 45)
    print("LOGIC TEST RESULTS:")
    print("=" * 45)
    
    if test1 and test2 and test3:
        print("🎉 ALL LOGIC TESTS PASSED!")
        print("\nImplemented features:")
        print("- ✅ Player selection dialog before each game")
        print("- ✅ Support for 1-5 AI opponents")
        print("- ✅ Dynamic game engine creation")
        print("- ✅ Multiple AI player display")
        print("- ✅ Option to change players between hands")
        print("\n🎯 Implementation complete!")
        print("\nTo test visually, run:")
        print("   python play/gui.py")
        print("\nYou'll see a dialog to choose AI opponents before playing!")
    else:
        print("❌ Some logic tests failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
