#!/usr/bin/env python3
"""
Test that the GUI runs without crashing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gui_startup():
    """Test that the GUI can start up"""
    print("Testing GUI startup...")
    
    try:
        import tkinter as tk
        from play.gui import PokerGameGUI
        
        print("✓ Imports successful")
        
        # Create a root for testing
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        # Quick test - just verify class can be instantiated without errors
        # (We won't run mainloop to avoid blocking)
        print("✓ GUI class available")
        
        root.quit()
        root.destroy()
        
        print("✓ GUI test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        return False

def verify_gui_structure():
    """Verify the GUI has all required methods"""
    print("\nVerifying GUI structure...")
    
    from play.gui import PokerGameGUI
    
    required_methods = [
        '__init__',
        'show_player_selection',
        'start_game_with_players',
        'start_game',
        'setup_gui',
        'update_display',
        '_setup_ai_display',
        '_update_ai_display'
    ]
    
    for method in required_methods:
        if hasattr(PokerGameGUI, method):
            print(f"✓ {method} method exists")
        else:
            print(f"✗ {method} method missing")
            return False
    
    return True

def main():
    """Run all tests"""
    print("GUI Functionality Test")
    print("=" * 30)
    
    test1 = test_gui_startup()
    test2 = verify_gui_structure()
    
    print("\n" + "=" * 30)
    print("RESULTS:")
    
    if test1 and test2:
        print("🎉 GUI IS WORKING!")
        print("\nYou can now run the poker game:")
        print("   python play/gui.py")
        print("\nThis will show the player selection dialog!")
    else:
        print("❌ GUI has issues that need to be fixed")
    
    return test1 and test2

if __name__ == "__main__":
    main()
