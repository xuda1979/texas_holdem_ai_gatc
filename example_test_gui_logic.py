#!/usr/bin/env python3
"""
Test the critical fix for infinite recursion in GUI
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_gui_hand_end_logic():
    """Test the _handle_hand_end method doesn't cause infinite recursion"""
    print("Testing GUI hand end logic...")

    # Import the GUI class
    from play.gui import PokerGameGUI

    # Check if the _show_next_hand_option method exists
    if hasattr(PokerGameGUI, "_show_next_hand_option"):
        print("✓ _show_next_hand_option method exists")
    else:
        print("✗ _show_next_hand_option method not found")
        return False

    # Check if the _start_next_hand method exists
    if hasattr(PokerGameGUI, "_start_next_hand"):
        print("✓ _start_next_hand method exists")
    else:
        print("✗ _start_next_hand method not found")
        return False

    # Read the _handle_hand_end method to ensure it doesn't directly call start_game
    with open("play/gui.py") as f:
        content = f.read()

    # Find the _handle_hand_end method
    import re

    match = re.search(
        r"def _handle_hand_end\(self\):(.*?)(?=\n    def|\n\nif __name__|$)", content, re.DOTALL
    )
    if match:
        method_content = match.group(1)
        # Check that it doesn't directly call self.start_game()
        if "self.start_game()" in method_content:
            print("✗ _handle_hand_end still contains direct call to self.start_game()")
            return False
        else:
            print("✓ _handle_hand_end no longer directly calls self.start_game()")

        # Check that it calls _show_next_hand_option instead
        if "_show_next_hand_option" in method_content:
            print("✓ _handle_hand_end calls _show_next_hand_option")
        else:
            print("✗ _handle_hand_end doesn't call _show_next_hand_option")
            return False
    else:
        print("✗ Could not find _handle_hand_end method")
        return False

    return True


if __name__ == "__main__":
    success = test_gui_hand_end_logic()
    if success:
        print("\n✓ GUI infinite recursion fix verified!")
    else:
        print("\n✗ GUI infinite recursion fix failed!")
        sys.exit(1)
