from __future__ import annotations

import eval7

_VALID_RANKS = set("23456789TJQKA")
_VALID_SUITS = set("cdhs")


def _normalize(card: str) -> str:
    """Normalize a card like 'As', 'td', 'QH' -> 'As', validate rank/suit."""
    c = card.strip()
    if len(c) != 2:
        raise ValueError(f"Card must be 2 chars like 'As', got {card!r}")
    r, s = c[0].upper(), c[1].lower()
    if r not in _VALID_RANKS or s not in _VALID_SUITS:
        raise ValueError(f"Invalid card {card!r}")
    return r + s


def _to_eval7(card: str) -> eval7.Card:
    return eval7.Card(_normalize(card))


def evaluate_hand(hole: list[str], board: list[str]) -> int:
    """
    Evaluate a player's best 5-card hand score given 2 hole + up to 5 board cards.
    Returns eval7's integer score (higher is better).
    """
    all_cards = [_to_eval7(c) for c in (hole + board)]
    if not (5 <= len(all_cards) <= 7):
        raise ValueError("evaluate_hand expects 5 to 7 total cards")
    # Deduplicate safety
    if len({str(c) for c in all_cards}) != len(all_cards):
        raise ValueError("Duplicate cards detected in evaluate_hand")
    return eval7.evaluate(all_cards)


def best_of(players_hole: dict[int, list[str]], board: list[str]) -> list[int]:
    """
    Return list of winning player ids (handles ties) for a given board.
    Validates cards and rejects duplicates across all players + board.
    """
    board_norm = [_normalize(c) for c in board]
    seen = set(board_norm)
    scores: dict[int, int] = {}

    for pid, hole in players_hole.items():
        if len(hole) != 2:
            raise ValueError(f"Player {pid} must have exactly 2 hole cards")
        hole_norm = [_normalize(c) for c in hole]
        for c in hole_norm:
            if c in seen:
                raise ValueError(f"Duplicate card detected: {c}")
            seen.add(c)
        scores[pid] = eval7.evaluate([eval7.Card(c) for c in (hole_norm + board_norm)])

    max_score = max(scores.values())
    return [pid for pid, sc in scores.items() if sc == max_score]
