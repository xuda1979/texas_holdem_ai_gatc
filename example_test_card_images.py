#!/usr/bin/env python3
"""
Test card image loading in the GUI
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk

from play.gui import PokerGameGUI


def test_card_images():
    """Test that card images load properly in the GUI"""
    print("Testing card image loading...")

    try:
        # Create a hidden root window for testing
        root = tk.Tk()
        root.withdraw()

        # Create GUI instance (but don't show it)
        gui = PokerGameGUI(root)

        # Check how many card images were loaded
        loaded_count = len([k for k, v in gui.card_images.items() if v is not None])
        total_cards = len(gui.card_images)

        print("✓ GUI created successfully")
        print(f"✓ Card images loaded: {loaded_count}/{total_cards}")

        # List some example loaded cards
        loaded_cards = [k for k, v in gui.card_images.items() if v is not None][:10]
        print(f"✓ Example loaded cards: {loaded_cards}")

        # Check if high cards are loaded
        high_cards = ['As', 'Kh', 'Qd', 'Jc', 'Ts']
        high_cards_loaded = [card for card in high_cards if card in gui.card_images and gui.card_images[card] is not None]
        print(f"✓ High cards loaded: {high_cards_loaded}")

        root.destroy()

        if loaded_count >= 50:  # Should have almost all 52 cards
            print("🎉 Card images are working perfectly!")
            return True
        else:
            print(f"⚠️  Only {loaded_count} cards loaded. Some may be missing or corrupted.")
            return False

    except Exception as e:
        print(f"✗ Card image test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_card_images()
    if success:
        print("\n✅ Card images are ready! You can now run:")
        print("   python play/gui.py")
        print("\nThe GUI will display actual card images instead of text!")
    else:
        print("\n❌ There may be issues with card image loading.")
