import os
import sys
import random
import time

import eval7

# ensure src on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from gatc_poker.handeval import evaluate_hand  # noqa: E402


def _random_hand() -> tuple[list[str], list[str]]:
    ranks = list("23456789TJQKA")
    suits = list("cdhs")
    deck = [r + s for r in ranks for s in suits]
    cards = random.sample(deck, 7)
    return cards[:2], cards[2:]


def test_rank_equivalence_vs_eval7() -> None:
    random.seed(0)
    iterations = 200_000
    start = time.time()
    for _ in range(iterations):
        hole, board = _random_hand()
        score1 = evaluate_hand(hole, board)
        score2 = eval7.evaluate([eval7.Card(c) for c in hole + board])
        assert score1 == score2
    duration = time.time() - start
    per100k = duration / iterations * 100_000
    print(f"hand eval time per 100k: {per100k:.2f}s")


def test_two_pair_kicker_ordering() -> None:
    board = ["Ah", "Ac", "Kd", "7s", "2d"]
    better = ["Kh", "Qh"]
    worse = ["Kc", "Jh"]
    assert evaluate_hand(better, board) > evaluate_hand(worse, board)
    # identical kickers tie
    tie1 = ["Kh", "Qh"]
    tie2 = ["Kc", "Qd"]
    assert evaluate_hand(tie1, board) == evaluate_hand(tie2, board)
