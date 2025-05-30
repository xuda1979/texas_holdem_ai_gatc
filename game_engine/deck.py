
import random

class Deck:
    def __init__(self):
        self.cards = self.generate_deck()
    
    def generate_deck(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return [(rank, suit) for suit in suits for rank in ranks]
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, num_cards):
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards. Only {len(self.cards)} cards remaining in the deck.")
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:] # This slicing handles num_cards=0 correctly too
        return dealt_cards
