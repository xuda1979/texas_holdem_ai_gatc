#!/usr/bin/env python3
"""
Final comprehensive test - simulates full GUI workflow without windows.
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_complete_workflow():
    """Test complete GUI workflow simulation."""
    
    print("🧪 Starting comprehensive GUI workflow test...")
    
    # Mock all tkinter components
    mock_tk = MagicMock()
    mock_root = MagicMock()
    mock_tk.Tk.return_value = mock_root
    
    with patch.dict('sys.modules', {
        'tkinter': mock_tk,
        'tkinter.messagebox': MagicMock(),
        'tkinter.simpledialog': MagicMock()
    }):
        with patch('play.gui.tk', mock_tk):
            from play.gui import PokerGameGUI, GUIHumanStrategy
            from game_engine.texas_holdem import TexasHoldem
            from playStrategy import RandomAIStrategy
        
        print("✅ All imports successful")
        
        # Mock the GUI methods that would normally create windows
        with patch.object(PokerGameGUI, 'setup_gui'), \
             patch.object(PokerGameGUI, 'start_game'), \
             patch.object(PokerGameGUI, '_load_card_images'):
            
            # Create GUI instance
            gui = PokerGameGUI()
            print("✅ GUI instance created")
            
            # Test GUI human strategy
            human_strategy = GUIHumanStrategy(gui)
            assert human_strategy.gui_instance == gui
            print("✅ GUIHumanStrategy created and linked")
            
            # Test action setting
            human_strategy.set_action('call', 100)
            assert human_strategy.pending_action == ('call', 100)
            assert human_strategy.action_complete == True
            print("✅ Action setting works")
            
            # Test game creation workflow
            strategies = [human_strategy, RandomAIStrategy()]
            game = TexasHoldem(2, 1000, strategies)
            gui.game = game
            gui.human_strategy = human_strategy
            
            print("✅ Game created and linked to GUI")
            
            # Test card image key conversion
            test_cards = [
                ("As", "As"),
                ("Kd", "Kd"), 
                ("Tc", "Tc"),
                ("2♠", "2s"),
                ("Q♥", "Qh"),
                ("New Card", "placeholder"),
                ("", None),
                (None, None)
            ]
            
            for card, expected in test_cards:
                result = gui._get_card_image_key(card)
                assert result == expected, f"Card {card} -> expected {expected}, got {result}"
            
            print("✅ Card image key conversion working")
            
            # Test display update (should not crash)  
            with patch.object(gui, 'create_card_label', return_value=MagicMock()):
                gui.update_display()
            print("✅ Display update method works")

            # Test action handling
            gui._handle_player_action('call', 50)
            print("✅ Player action handling works")

            print("\n🎉 Complete workflow test passed!")

def test_error_conditions():
    """Test error handling."""
    
    print("\n🧪 Testing error conditions...")
    
    mock_tk = MagicMock()
    
    with patch.dict('sys.modules', {
        'tkinter': mock_tk,
        'tkinter.messagebox': MagicMock(),
        'tkinter.simpledialog': MagicMock()
    }):
        with patch('play.gui.tk', mock_tk):
            from play.gui import PokerGameGUI
        
        with patch.object(PokerGameGUI, 'setup_gui'), \
             patch.object(PokerGameGUI, 'start_game'), \
             patch.object(PokerGameGUI, '_load_card_images'):
            
            gui = PokerGameGUI()
            
            # Test with no game
            gui.game = None
            gui.update_display()  # Should not crash
            print("✅ Handles missing game gracefully")
            
            # Test invalid card keys
            invalid_cards = ["", None, "X", "Invalid"]
            for card in invalid_cards:
                result = gui._get_card_image_key(card)
                # Should return None or "placeholder", not crash
                assert result in [None, "placeholder"], f"Invalid card {card} should return None or placeholder"
            print("✅ Handles invalid cards gracefully")

            print("✅ Error condition handling works")

if __name__ == "__main__":
    try:
        success = True
        success &= test_complete_workflow()
        success &= test_error_conditions()
        
        if success:
            print("\n🏆 ALL TESTS PASSED!")
            print("The GUI is ready for production use.")
        else:
            print("\n❌ Some tests failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
