"""Utilities for managing a standard 52-card deck.

The training engine historically reimplemented deck management logic directly
inside :mod:`poker_ai.engine.texas_holdem`.  As the code base grew that
monolithic approach made it increasingly difficult to reason about state
changes, add deterministic shuffles for tests or reuse the behaviour in other
modes such as simplified simulators.  The :class:`DeckManager` class defined
here encapsulates the operations for creating, shuffling and dealing cards while
keeping backwards compatibility with the rest of the project.  The class mutates
its internal :class:`CardDeck` instance in place so existing references (e.g. in
tests that inspect ``game.rules.deck.cards``) remain valid after resets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable, List


class CardDeck(list):
    """Simple list subclass exposing a ``cards`` attribute for legacy tests."""

    @property
    def cards(self) -> List[str]:
        return list(self)

__all__ = ["SUITS", "RANKS", "DeckManager", "CardDeck"]

SUITS: List[str] = ["h", "d", "c", "s"]
"""Card suit symbols used across the engine."""

RANKS: List[str] = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
"""Card rank symbols ordered from lowest to highest."""


@dataclass
class DeckManager:
    """Manage a deck of cards with deterministic shuffling support."""

    rng: random.Random = field(default_factory=random.Random)
    cards: CardDeck = field(default_factory=CardDeck)

    def __post_init__(self) -> None:
        # When ``cards`` is provided we copy the sequence to avoid mutating the
        # caller's list.  Otherwise we initialise a fresh ordered deck.
        if self.cards:
            # ``cards`` may originate from another :class:`DeckManager`.
            self.cards = CardDeck(self.cards)
        else:
            self.reset()

    # ------------------------------------------------------------------
    # Deck construction helpers
    # ------------------------------------------------------------------
    def build_fresh_deck(self) -> List[str]:
        """Return a new ordered deck without mutating ``self.cards``."""

        return [rank + suit for suit in SUITS for rank in RANKS]

    def reset(self) -> None:
        """Restore ``self.cards`` to an ordered, unshuffled state."""

        fresh_deck = self.build_fresh_deck()
        if isinstance(self.cards, CardDeck):
            # Mutate in place so existing references (e.g. tests inspecting
            # ``rules.deck``) continue to observe the same list object.
            self.cards[:] = fresh_deck
        else:
            self.cards = CardDeck(fresh_deck)

    # ------------------------------------------------------------------
    # Deck manipulation
    # ------------------------------------------------------------------
    def shuffle(self) -> None:
        """Shuffle the deck using the configured random number generator."""

        self.rng.shuffle(self.cards)

    def draw(self, count: int = 1) -> List[str]:
        """Remove and return ``count`` cards from the top of the deck."""

        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self.cards):
            raise ValueError("Cannot draw more cards than remain in the deck.")

        drawn: List[str] = []
        for _ in range(count):
            drawn.append(self.cards.pop())
        return drawn

    def burn(self) -> str:
        """Remove and return a single burn card from the deck."""

        if not self.cards:
            raise ValueError("The deck is empty; cannot burn a card.")
        return self.cards.pop()

    def deal_hole_cards(
        self,
        hands: List[List[str]],
        active_players: Iterable[bool],
        *,
        cards_per_player: int = 2,
    ) -> None:
        """Deal ``cards_per_player`` cards to each active player."""

        for _ in range(cards_per_player):
            for idx, is_active in enumerate(active_players):
                if not is_active:
                    continue
                if not self.cards:
                    raise ValueError("The deck is empty; cannot deal more cards.")
                hands[idx].append(self.cards.pop())

    # ------------------------------------------------------------------
    # Copy helpers
    # ------------------------------------------------------------------
    def clone(self) -> "DeckManager":
        """Return a deep copy preserving RNG state and remaining cards."""

        new_rng = random.Random()
        new_rng.setstate(self.rng.getstate())
        return DeckManager(rng=new_rng, cards=CardDeck(self.cards))
