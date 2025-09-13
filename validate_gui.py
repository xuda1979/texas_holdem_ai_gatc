#!/usr/bin/env python3
"""
Simple GUI validation script - test without running mainloop.
"""

import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_gui_without_mainloop() -> bool:
    """Test GUI creation without running mainloop."""
    
    # Mock tkinter to prevent window creation
    with patch('tkinter.Tk') as mock_tk:
        mock_root = Mock()
        mock_tk.return_value = mock_root
        
        # Mock mainloop to prevent it from running
        mock_root.mainloop = Mock()
        
        # Also mock the GUI setup methods to prevent them from running
        with patch('play.gui.PokerGameGUI.setup_gui'), \
             patch('play.gui.PokerGameGUI.start_game'), \
             patch('play.gui.PokerGameGUI._load_card_images'):
            
            # Now we can safely import and create the GUI
            from play.gui import PokerGameGUI, GUIHumanStrategy
            
            print("✅ GUI classes imported successfully")
            
            # Test GUIHumanStrategy
            mock_gui = Mock()
            GUIHumanStrategy(mock_gui)  # Test creation without assigning to unused variable
            print("✅ GUIHumanStrategy created successfully")
            
            # Test card image key conversion
            gui = PokerGameGUI()
            
            # Test the _get_card_image_key method
            test_cases = [
                ("As", "As"),
                ("Kh", "Kh"),
                ("2♠", "2s"),
                ("J♥", "Jh"),
                ("New Card", "placeholder"),
                ("", None),
                (None, None),
            ]
            
            for input_card, expected in test_cases:
                result = gui._get_card_image_key(input_card)
                assert result == expected, f"Expected {expected}, got {result} for input {input_card}"
            
            print("✅ Card image key conversion working correctly")
            
            # Verify that mainloop was mocked and not actually called
            assert mock_root.mainloop.called, "Mainloop should have been called (but mocked)"
            print("✅ GUI initialization completed without opening windows")
            
            return True

if __name__ == "__main__":
    try:
        test_gui_without_mainloop()
        print("\n🎉 GUI validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ GUI validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
