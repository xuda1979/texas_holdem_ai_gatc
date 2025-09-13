#!/usr/bin/env python3
"""Simple test to verify GUI loads without displaying it."""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_gui_load():
    """Test that GUI loads properly."""
    try:
        from play.gui import PokerGameGUI

        print("✓ GUI module imports successfully")

        # Create GUI instance (but don't run mainloop)
        gui = PokerGameGUI()
        print("✓ GUI instance created successfully")

        # Check card images
        print(f"✓ {len(gui.card_images)} card images loaded")

        # Test specific cards
        test_cards = ["As", "Kh", "Qd", "Jc"]
        for card in test_cards:
            if card in gui.card_images:
                print(f"  ✓ {card} image loaded")
            else:
                print(f"  ✗ {card} image not found")

        # Cleanup
        gui.root.destroy()
        print("✓ Test completed successfully")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_gui_load()
