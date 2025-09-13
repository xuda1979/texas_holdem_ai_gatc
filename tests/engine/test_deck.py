import os
import sys
import random
from collections import Counter

import numpy as np
import pytest

stats = pytest.importorskip("scipy.stats")
chisquare = stats.chisquare

# ensure src is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.deck import Deck  # noqa: E402


def test_no_duplicates() -> None:
    deck = Deck()
    deck.shuffle()
    dealt = deck.deal(52)
    assert len(dealt) == 52
    assert len(set(dealt)) == 52


def test_correct_counts_preflop_flop_turn_river() -> None:
    deck = Deck()
    deck.shuffle()
    players = 4
    preflop = []
    for _ in range(players):
        preflop.extend(deck.deal(2))
    assert len(deck.cards) == 52 - players * 2
    flop = deck.deal(3)
    assert len(deck.cards) == 52 - players * 2 - 3
    turn = deck.deal(1)
    assert len(deck.cards) == 52 - players * 2 - 4
    river = deck.deal(1)
    assert len(deck.cards) == 52 - players * 2 - 5
    # ensure all cards unique
    all_cards = preflop + flop + turn + river
    assert len(set(all_cards)) == len(all_cards)


def test_shuffle_uniformity_chi2() -> None:
    random.seed(0)
    samples = 5200
    first_cards = []
    base_deck = Deck().cards
    for _ in range(samples):
        d = Deck()
        d.shuffle()
        first_cards.append(d.deal(1)[0])
    counts = Counter(first_cards)
    observed = np.array([counts.get(card, 0) for card in base_deck])
    expected = np.full(len(base_deck), samples / len(base_deck))
    chi2, p = chisquare(observed, expected)
    assert p > 0.01
