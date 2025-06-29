#!/usr/bin/env python3
"""Test script to verify card image functionality in the GUI."""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

def test_card_image_loading():
    """Test that card images can be loaded."""
    try:
        from play.gui import PokerGameGUI
        print("✓ GUI module imports successfully")
        
        # Create GUI instance (this will load card images)
        gui = PokerGameGUI()
        print("✓ GUI instance created successfully")
        
        # Check if card images were loaded
        if gui.card_images:
            print(f"✓ {len(gui.card_images)} card images loaded")
            
            # Test a few specific cards
            test_cards = ['As', 'Kh', 'Qd', 'Jc', 'Ts', '9h', '2s']
            for card in test_cards:
                if card in gui.card_images:
                    print(f"  ✓ {card} image loaded")
                else:
                    print(f"  ✗ {card} image missing")
        else:
            print("✗ No card images loaded")
            
        # Check card back image
        if gui.card_back_image:
            print("✓ Card back image created")
        else:
            print("✗ Card back image not created")
            
        # Test card label creation
        try:
            import tkinter as tk
            test_frame = tk.Frame(gui.root)
            card_label = gui.create_card_label(test_frame, 'As')
            print("✓ Card label creation works")
        except Exception as e:
            print(f"✗ Card label creation failed: {e}")
            
        gui.root.destroy()
        print("✓ GUI cleanup successful")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        if "PIL" in str(e):
            print("  Note: Install Pillow with: pip install Pillow")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("Testing card image functionality...")
    test_card_image_loading()
    print("Test completed.")
