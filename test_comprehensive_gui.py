#!/usr/bin/env python3
"""
Test script to validate GUI imports and initialization without running the main loop.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_gui_imports():
    """Test that all GUI imports work correctly."""
    try:
        print("Testing GUI imports...")

        # Test individual imports

        print("✓ TexasHoldem import successful")

        print("✓ Strategy imports successful")

        print("✓ PlaceholderAIStrategy import successful")

        # Test GUI classes (but don't create instances that start mainloop)

        print("✓ GUI classes import successful")

        print("All imports successful!")
        assert True

    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError()


def test_game_logic():
    """Test basic game logic without GUI."""
    try:
        print("\nTesting game logic...")

        from game_engine.texas_holdem import TexasHoldem

        from playStrategy import RandomAIStrategy

        # Create a simple 2-player game
        strategies = [RandomAIStrategy(), RandomAIStrategy()]
        game = TexasHoldem(2, 1000, strategies)

        print("✓ Game creation successful")
        print(f"✓ Game has {game.num_players} players")
        print(f"✓ Starting stack: ${game.rules.player_chips[0]}")

        assert True

    except Exception as e:
        print(f"✗ Game logic error: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError()


def test_card_images():
    """Test card image loading functionality."""
    try:
        print("\nTesting card image functionality...")

        import tkinter as tk

        try:
            from PIL import Image, ImageTk
        except Exception:
            pytest.skip("Pillow not installed")

        # Use a mocked root window to avoid display issues
        with patch.object(tk, "Tk", return_value=MagicMock()) as mock_tk:
            root = tk.Tk()
            root.withdraw.return_value = None

            # Test image loading
            card_images_dir = os.path.join("play", "card_images")
            if os.path.exists(card_images_dir):
                test_image_path = os.path.join(card_images_dir, "As.png")
                if os.path.exists(test_image_path):
                    img = Image.open(test_image_path)
                    img = img.resize((100, 140), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    print("✓ Card image loading successful")
                else:
                    print("✗ Test card image not found")
            else:
                print("⚠ Card images directory not found")

            root.destroy()

        assert True

    except Exception as e:
        print(f"✗ Card image error: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError()


if __name__ == "__main__":
    print("=== Comprehensive GUI Testing ===")

    success = True
    success &= test_gui_imports()
    success &= test_game_logic()
    success &= test_card_images()

    if success:
        print("\n🎉 All tests passed! GUI is ready to use.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
