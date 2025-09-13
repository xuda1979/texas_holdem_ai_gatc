#!/usr/bin/env python3
"""
Display information about the created card images
"""
import os

from PIL import Image


def show_card_info() -> None:
    """Show information about a sample card image"""
    sample_card = "play/card_images/As.png"  # Ace of Spades

    if not os.path.exists(sample_card):
        print("Sample card not found")
        return

    try:
        # Load and get info about the image
        img = Image.open(sample_card)
        width, height = img.size
        mode = img.mode
        size = os.path.getsize(sample_card)

        print("Sample Card: Ace of Spades (As.png)")
        print(f"  Dimensions: {width} x {height} pixels")
        print(f"  Color mode: {mode}")
        print(f"  File size: {size} bytes")
        print("  Format: PNG")

        # Check if it's a reasonable card size
        if width >= 50 and height >= 70:
            print("  ✅ Good size for display")
        else:
            print("  ⚠️  Might be too small for clear display")

        print("\nAll 52 cards follow the same format:")
        print("  - Named as: [rank][suit].png (e.g., 2h.png, Kc.png, As.png)")
        print("  - Ranks: 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A")
        print("  - Suits: h (hearts), d (diamonds), c (clubs), s (spades)")

    except Exception as e:
        print(f"Error reading sample card: {e}")

def list_card_organization() -> None:
    """Show how the cards are organized"""
    print("Card Organization:")
    print("=" * 30)

    suits = [('h', 'Hearts ♥'), ('d', 'Diamonds ♦'), ('c', 'Clubs ♣'), ('s', 'Spades ♠')]
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    for suit_code, suit_name in suits:
        cards_in_suit = [f"{rank}{suit_code}" for rank in ranks]
        print(f"{suit_name}: {', '.join(cards_in_suit)}")

if __name__ == "__main__":
    show_card_info()
    print()
    list_card_organization()

    print("\n🎯 Ready to use!")
    print("The GUI will now display these card images when you run:")
    print("   python play/gui.py")
