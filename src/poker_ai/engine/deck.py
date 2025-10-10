from __future__ import annotations

import random
from typing import List, Optional, Tuple

Card = Tuple[str, str]


class Deck:
    """Representation of a standard 52-card deck with optional deterministic RNG."""

    def __init__(self, *, seed: Optional[int] = None, rng: Optional[random.Random] = None):
        if seed is not None and rng is not None:
            raise ValueError("Pass either 'seed' or 'rng', not both.")
        self._rng = rng if rng is not None else random.Random()
        if seed is not None:
            self._rng.seed(seed)
        self.cards: List[Card] = self.generate_deck()

    def generate_deck(self) -> List[Card]:
        suits = ["hearts", "diamonds", "clubs", "spades"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return [(rank, suit) for suit in suits for rank in ranks]

    def shuffle(self) -> None:
        self._rng.shuffle(self.cards)

    def deal(self, num_cards: int) -> List[Card]:
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Restore the deck to an ordered state and optionally reseed the RNG."""

        self.cards = self.generate_deck()
        if seed is not None:
            self._rng.seed(seed)

    def set_seed(self, seed: int) -> None:
        """Reseed the deck's RNG without altering the remaining cards."""

        self._rng.seed(seed)

    def copy(self) -> "Deck":
        new_rng = random.Random()
        new_rng.setstate(self._rng.getstate())
        new_deck = Deck(rng=new_rng)
        new_deck.cards = self.cards.copy()
        return new_deck

    @property
    def remaining(self) -> int:
        return len(self.cards)
