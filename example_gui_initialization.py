#!/usr/bin/env python3
"""
Test script to verify GUI can be started without infinite recursion
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from play.gui import PokerGameGUI

def test_gui_initialization():
    """Test that GUI can be initialized without crashing"""
    print("Testing GUI initialization...")
    
    try:
        # Create a root window (but don't show it)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Make sure we're running in headless mode for testing
        try:
            root.tk.call('wm', 'iconify', root)
        except:
            pass
        
        # Try to create the GUI instance
        gui = PokerGameGUI(root)
        print("✓ GUI initialized successfully")
        
        # Check if the game engine is properly set up
        print(f"✓ Game engine created with {gui.game_engine.num_players} players")
        print(f"✓ Player chips: {gui.game_engine.rules.player_chips}")
        
        # Test that we can access key GUI components
        print(f"✓ Main canvas created: {gui.canvas is not None}")
        print(f"✓ Button frame created: {gui.button_frame is not None}")
        
        # Close the window
        root.quit()
        root.destroy()
        
        return True
    except Exception as e:
        print(f"✗ GUI initialization failed: {e}")
        try:
            root.quit()
            root.destroy()
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_gui_initialization()
    if success:
        print("\n✓ GUI initialization test passed!")
    else:
        print("\n✗ GUI initialization test failed!")
        sys.exit(1)
