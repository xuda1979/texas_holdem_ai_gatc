import os
import random
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (project_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.deck import Deck  # noqa: E402


def test_generate_deck_unique() -> None:
    deck = Deck()
    assert len(deck.cards) == 52
    assert len(set(deck.cards)) == 52


def test_shuffle_changes_order() -> None:
    deck = Deck()
    original = deck.cards.copy()
    random.seed(0)
    deck.shuffle()
    assert deck.cards != original


def test_deal_reduces_deck_size() -> None:
    deck = Deck()
    deck.shuffle()
    dealt = deck.deal(5)
    assert len(dealt) == 5
    assert len(deck.cards) == 47
