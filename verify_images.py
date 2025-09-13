#!/usr/bin/env python3
"""
Verify that all poker card images are present and properly named
"""
import os


def verify_card_images():
    """Verify all 52 card images are present with correct names"""
    print("Verifying poker card images...")
    print("=" * 40)

    suits = ['h', 'd', 'c', 's']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    card_dir = "play/card_images"

    if not os.path.exists(card_dir):
        print(f"❌ Card directory {card_dir} does not exist!")
        return False

    missing_cards = []
    existing_cards = []
    file_sizes = {}

    for suit in suits:
        for rank in ranks:
            card_name = f"{rank}{suit}"
            file_path = os.path.join(card_dir, f"{card_name}.png")

            if os.path.exists(file_path):
                existing_cards.append(card_name)
                file_size = os.path.getsize(file_path)
                file_sizes[card_name] = file_size

                # Check if file is reasonable size (not empty, not too large)
                if file_size < 100:
                    print(f"⚠️  {card_name}.png is very small ({file_size} bytes)")
                elif file_size > 50000:
                    print(f"⚠️  {card_name}.png is very large ({file_size} bytes)")
            else:
                missing_cards.append(card_name)

    print(f"✅ Found {len(existing_cards)}/52 card images")

    if missing_cards:
        print(f"❌ Missing {len(missing_cards)} cards:")
        print(f"   {', '.join(missing_cards[:10])}{'...' if len(missing_cards) > 10 else ''}")
        return False

    # Show some examples of what we have
    example_cards = ['2h', '7d', 'Tc', 'Jh', 'Qs', 'Kc', 'As']
    print("\nExample cards with sizes:")
    for card in example_cards:
        if card in file_sizes:
            print(f"   {card}.png: {file_sizes[card]} bytes")

    # Check for duplicates or unexpected files
    all_files = [f for f in os.listdir(card_dir) if f.endswith('.png')]
    expected_files = [f"{rank}{suit}.png" for suit in suits for rank in ranks]

    unexpected_files = [f for f in all_files if f not in expected_files]
    if unexpected_files:
        print(f"\nUnexpected files found: {unexpected_files}")

    print("\n🎉 SUCCESS! All 52 poker card images are present and ready!")
    return True

def check_poker_table_background():
    """Check if poker table background exists"""
    bg_path = "play/assets/poker_table_background.png"
    if os.path.exists(bg_path):
        size = os.path.getsize(bg_path)
        print(f"✅ Poker table background found ({size} bytes)")
        return True
    else:
        print(f"⚠️  Poker table background not found at {bg_path}")
        return False

def main():
    """Main verification function"""
    print("Poker Image Assets Verification")
    print("=" * 50)

    cards_ok = verify_card_images()
    print()
    bg_ok = check_poker_table_background()

    print("\n" + "=" * 50)
    print("FINAL RESULT:")
    print("=" * 50)

    if cards_ok:
        print("🎉 EXCELLENT! All poker card images are ready!")
        print("\nYour Texas Hold'em AI now has:")
        print("   ✅ All 52 poker card images")
        if bg_ok:
            print("   ✅ Poker table background")
        else:
            print("   ⚠️  Basic poker table background")

        print("\n🚀 Ready to play! Run the GUI:")
        print("   python play/gui.py")
        print("\nThe game will now show actual card images instead of text!")

    else:
        print("❌ Some card images are missing. Please run:")
        print("   python setup_card_images.py")

if __name__ == "__main__":
    main()
