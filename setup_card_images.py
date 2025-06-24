#!/usr/bin/env python3
"""
Download and create poker card images for the Texas Hold'em AI
"""
import os
import sys
import urllib.request
import zipfile
from PIL import Image, ImageDraw, ImageFont
import requests

def create_simple_card_images():
    """Create simple text-based card images if we can't download fancy ones"""
    print("Creating simple text-based card images...")
    
    suits = ['h', 'd', 'c', 's']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    # Unicode symbols for suits
    suit_symbols = {
        'h': '♥',  # hearts
        'd': '♦',  # diamonds  
        'c': '♣',  # clubs
        's': '♠'   # spades
    }
    
    # Colors for suits
    suit_colors = {
        'h': (255, 0, 0),    # red
        'd': (255, 0, 0),    # red
        'c': (0, 0, 0),      # black
        's': (0, 0, 0)       # black
    }
    
    card_dir = "play/card_images"
    os.makedirs(card_dir, exist_ok=True)
    
    created_count = 0
    
    for suit in suits:
        for rank in ranks:
            card_name = f"{rank}{suit}"
            file_path = os.path.join(card_dir, f"{card_name}.png")
            
            # Skip if file already exists
            if os.path.exists(file_path):
                print(f"Skipping {card_name}.png (already exists)")
                continue
            
            # Create a simple card image
            width, height = 120, 180
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Draw border
            draw.rectangle([2, 2, width-3, height-3], outline='black', width=2)
            
            # Try to use a decent font
            try:
                font_large = ImageFont.truetype("arial.ttf", 36)
                font_small = ImageFont.truetype("arial.ttf", 24)
            except:
                try:
                    font_large = ImageFont.truetype("helvetica.ttf", 36)
                    font_small = ImageFont.truetype("helvetica.ttf", 24)
                except:
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()
            
            suit_color = suit_colors[suit]
            suit_symbol = suit_symbols[suit]
            
            # Draw rank in top-left
            draw.text((10, 10), rank, fill=suit_color, font=font_large)
            
            # Draw suit in top-left under rank
            draw.text((10, 50), suit_symbol, fill=suit_color, font=font_small)
            
            # Draw large suit symbol in center
            center_x, center_y = width // 2, height // 2
            draw.text((center_x - 20, center_y - 20), suit_symbol, fill=suit_color, font=font_large)
            
            # Draw rank in bottom-right (rotated)
            draw.text((width - 40, height - 50), rank, fill=suit_color, font=font_large)
            
            # Draw suit in bottom-right
            draw.text((width - 40, height - 90), suit_symbol, fill=suit_color, font=font_small)
            
            # Save the image
            image.save(file_path)
            print(f"Created {card_name}.png")
            created_count += 1
    
    print(f"\nCreated {created_count} new card images")
    return created_count

def download_fancy_card_deck():
    """Try to download a nice card deck from a public source"""
    print("Attempting to download fancy card deck...")
    
    # Try Byron Knoll's card deck (public domain)
    card_deck_url = "https://github.com/hayeah/playing-cards-assets/archive/master.zip"
    
    try:
        print(f"Downloading from {card_deck_url}")
        response = requests.get(card_deck_url, timeout=30)
        response.raise_for_status()
        
        # Save the zip file
        zip_path = "temp_cards.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract and organize cards
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_cards")
        
        # Look for PNG files in the extracted directory
        extracted_dir = "temp_cards"
        card_files = []
        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                if file.lower().endswith('.png') and len(file) == 6:  # e.g., "2h.png"
                    card_files.append(os.path.join(root, file))
        
        if card_files:
            print(f"Found {len(card_files)} card files")
            card_dir = "play/card_images"
            os.makedirs(card_dir, exist_ok=True)
            
            copied_count = 0
            for src_file in card_files:
                filename = os.path.basename(src_file)
                dst_file = os.path.join(card_dir, filename)
                
                if not os.path.exists(dst_file):
                    import shutil
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {filename}")
                    copied_count += 1
            
            print(f"Copied {copied_count} card images")
            
            # Cleanup
            import shutil
            shutil.rmtree("temp_cards", ignore_errors=True)
            os.remove(zip_path)
            
            return copied_count > 0
        
    except Exception as e:
        print(f"Failed to download fancy cards: {e}")
        return False

def check_missing_cards():
    """Check what cards are missing"""
    suits = ['h', 'd', 'c', 's']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    card_dir = "play/card_images"
    if not os.path.exists(card_dir):
        return 52, []
    
    existing_cards = []
    missing_cards = []
    
    for suit in suits:
        for rank in ranks:
            card_name = f"{rank}{suit}"
            file_path = os.path.join(card_dir, f"{card_name}.png")
            
            if os.path.exists(file_path):
                existing_cards.append(card_name)
            else:
                missing_cards.append(card_name)
    
    print(f"Existing cards: {len(existing_cards)}/52")
    print(f"Missing cards: {len(missing_cards)}/52")
    
    if len(missing_cards) <= 10:  # Show missing cards if not too many
        print(f"Missing: {', '.join(missing_cards)}")
    
    return len(missing_cards), missing_cards

def main():
    """Main function to download/create poker card images"""
    print("Poker Card Image Setup")
    print("=" * 40)
    
    # Check current status
    missing_count, missing_cards = check_missing_cards()
    
    if missing_count == 0:
        print("✓ All 52 card images are already present!")
        return
    
    print(f"\nNeed to create {missing_count} card images...")
    
    # Install required packages
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✓ PIL (Pillow) is available")
    except ImportError:
        print("Installing Pillow for image creation...")
        os.system(f"{sys.executable} -m pip install Pillow")
        from PIL import Image, ImageDraw, ImageFont
    
    try:
        import requests
        print("✓ requests is available")
    except ImportError:
        print("Installing requests for downloading...")
        os.system(f"{sys.executable} -m pip install requests")
        import requests
    
    # Try to download fancy cards first
    success = False
    try:
        success = download_fancy_card_deck()
    except Exception as e:
        print(f"Download failed: {e}")
    
    # Fall back to creating simple cards
    if not success:
        print("\nFalling back to creating simple text-based cards...")
        create_simple_card_images()
    
    # Final check
    print("\n" + "=" * 40)
    missing_count, _ = check_missing_cards()
    
    if missing_count == 0:
        print("🎉 SUCCESS! All 52 card images are now available!")
        print("\nYou can now run the GUI with full card graphics:")
        print("  python play/gui.py")
    else:
        print(f"❌ Still missing {missing_count} cards. There may have been an error.")

if __name__ == "__main__":
    main()
