#!/usr/bin/env python3
"""
Complete test of the GUI card display functionality.
This tests card display in the actual game context.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_gui_with_game():
    """Test GUI with actual game state including cards."""
    try:
        from play.gui import PokerGameGUI
        from game_engine.texas_holdem import TexasHoldem
        from playStrategy import RandomAIStrategy
        
        print("✓ All modules imported successfully")
        
        # Create GUI
        gui = PokerGameGUI()
        print(f"✓ GUI created with {len(gui.card_images)} card images loaded")
        
        # Create a test game with 3 players (1 human + 2 AI)
        player_strategies = [None, RandomAIStrategy(), RandomAIStrategy()]  # Human strategy added later
        game = TexasHoldem(3, 10000, player_strategies)
        gui.game = game
        
        # Simulate some cards being dealt
        gui.game.rules.hands = [['As', 'Kh'], ['Qd', 'Jc'], ['Ts', '9h']]
        gui.game.rules.community_cards = ['Ah', 'Kd', 'Qs']
        gui.game.rules.pot = 150
        
        print("✓ Test game state created with:")
        print(f"  - Player hand: {gui.game.rules.hands[0]}")
        print(f"  - Community cards: {gui.game.rules.community_cards}")
        print(f"  - Pot: ${gui.game.rules.pot}")
        
        # Test card label creation for different cards
        import tkinter as tk
        test_frame = tk.Frame(gui.root)
        
        test_cards = ['As', 'Kh', 'Qd', 'Jc', 'Ts']
        for card in test_cards:
            card_label = gui.create_card_label(test_frame, card)
            if card in gui.card_images:
                print(f"  ✓ {card} label created with image")
            else:
                print(f"  ✗ {card} label created without image")
        
        # Test card back creation
        back_label = gui.create_card_label(test_frame, 'As', show_back=True)
        print("  ✓ Card back label created")
        
        # Cleanup
        gui.root.destroy()
        print("✓ GUI test completed successfully!")
        print("\nCard images are ready for display in the poker game!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui_with_game()
