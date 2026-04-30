from __future__ import annotations

import itertools
import random

import pytest

eval7 = pytest.importorskip("eval7")

from gatc_poker.handeval import best_of, evaluate_hand

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]


# ---------------------------------------------------------------------------
# Canonical hands covering all hand categories. Each entry is in descending
# strength order so their eval7 score should be strictly decreasing.
CANONICAL_HANDS: list[tuple[list[str], list[str], str]] = [
    (["As", "Ks"], ["Qs", "Js", "Ts", "2c", "3d"], "Straight Flush"),
    (["Ah", "Ad"], ["Ac", "As", "Kc", "2d", "3h"], "Quads"),
    (["As", "Kd"], ["Ah", "Ad", "Ks", "2c", "3h"], "Full House"),
    (["As", "9s"], ["Qs", "Js", "8s", "2c", "3d"], "Flush"),
    (["9c", "8d"], ["7h", "6s", "5d", "2c", "3c"], "Straight"),
    (["As", "Ad"], ["Ac", "Ks", "Qd", "2c", "3h"], "Trips"),
    (["As", "Ad"], ["Ks", "Kd", "2c", "3d", "4h"], "Two Pair"),
    (["As", "Ad"], ["Qs", "Jd", "2c", "3d", "4h"], "Pair"),
    (["As", "Kd"], ["2c", "3d", "7h", "9s", "Tc"], "High Card"),
]


def test_category_order() -> None:
    """Scores for canonical hands should be strictly decreasing."""
    scores = []
    seen_types = []
    for hole, board, expected in CANONICAL_HANDS:
        score = evaluate_hand(hole, board)
        scores.append(score)
        seen_types.append(eval7.handtype(score))
        assert seen_types[-1] == expected
    for a, b in zip(scores, scores[1:]):
        assert a > b, f"{a} !> {b}"


# ---------------------------------------------------------------------------
# Tie-breakers inside categories


def _cmp(h1: list[str], b1: list[str], h2: list[str], b2: list[str]) -> None:
    s1 = evaluate_hand(h1, b1)
    s2 = evaluate_hand(h2, b2)
    assert s1 > s2, (s1, s2)


def test_tie_breakers() -> None:
    # Straight flush 9-high beats 8-high
    _cmp(
        ["9s", "8s"],
        ["7s", "6s", "5s", "2c", "3d"],
        ["8s", "7s"],
        ["6s", "5s", "4s", "2c", "3d"],
    )
    # Quads Aces beat quads Kings (kicker ignored)
    _cmp(
        ["As", "Ah"],
        ["Ad", "Ac", "Kc", "2d", "3h"],
        ["Ks", "Kh"],
        ["Kd", "Kc", "As", "2d", "3h"],
    )
    # Full house Aces full of Kings beats Aces full of Queens
    _cmp(
        ["As", "Ah"],
        ["Ad", "Ks", "Kd", "2c", "3h"],
        ["As", "Ah"],
        ["Ad", "Qs", "Qd", "2c", "3h"],
    )
    # Flush A-high beats K-high flush
    _cmp(
        ["As", "9s"],
        ["Ts", "8s", "3c", "2d", "4h"],
        ["Ks", "9s"],
        ["Ts", "8s", "3c", "2d", "4h"],
    )
    # Straight 9-high beats 8-high
    _cmp(
        ["9c", "8d"],
        ["7h", "6s", "5d", "2c", "3c"],
        ["8c", "7d"],
        ["6h", "5s", "4d", "2c", "3c"],
    )
    # Trips Aces beat trips Kings
    _cmp(
        ["As", "Ah"],
        ["Ad", "Ks", "Qd", "2c", "3h"],
        ["Ks", "Kh"],
        ["Kc", "As", "Qd", "2c", "3h"],
    )
    # Two pair AA+KK beats KK+QQ
    _cmp(
        ["As", "Ah"],
        ["Ks", "Kd", "2c", "3d", "4h"],
        ["Ks", "Kh"],
        ["Qs", "Qd", "2c", "3d", "4h"],
    )
    # One pair Aces beats Kings
    _cmp(
        ["As", "Ah"],
        ["Qs", "Jd", "2c", "3d", "4h"],
        ["Ks", "Kh"],
        ["Qs", "Jd", "2c", "3d", "4h"],
    )
    # High card Ace-high beats King-high
    _cmp(
        ["As", "9d"],
        ["Qs", "Jc", "2d", "3h", "4h"],
        ["Ks", "9d"],
        ["Qs", "Jc", "2d", "3h", "4h"],
    )


# ---------------------------------------------------------------------------
# Monotonicity: adding board cards can only improve or keep a hand's score.


def test_monotonicity_random() -> None:
    rng = random.Random(0)
    for _ in range(200):  # ~200 random samples
        deck = DECK.copy()
        rng.shuffle(deck)
        hole = deck[:2]
        flop = deck[2:5]
        turn = flop + deck[5:6]
        river = turn + deck[6:7]
        s1 = evaluate_hand(hole, flop)
        s2 = evaluate_hand(hole, turn)
        s3 = evaluate_hand(hole, river)
        assert s1 <= s2 <= s3


# ---------------------------------------------------------------------------
# Exhaustive enumeration on a fixed river board compares eval results to
# best_of outcome to ensure agreement.


def test_exhaustive_enumeration_matches_best_of() -> None:
    board = ["As", "Kc", "Qd", "Jc", "7h"]
    deck = [c for c in DECK if c not in board]
    for cards in itertools.combinations(deck, 4):
        h1 = list(cards[:2])
        h2 = list(cards[2:])
        score1 = eval7.evaluate([eval7.Card(c) for c in h1 + board])
        score2 = eval7.evaluate([eval7.Card(c) for c in h2 + board])
        winners = best_of({0: h1, 1: h2}, board)
        if score1 > score2:
            assert winners == [0]
        elif score2 > score1:
            assert winners == [1]
        else:
            assert set(winners) == {0, 1}
