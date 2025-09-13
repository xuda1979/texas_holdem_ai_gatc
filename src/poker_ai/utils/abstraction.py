"""Simple action and hand abstraction utilities."""

HAND_BUCKETS = [
    {"ranks": ["A", "K"], "bucket": 0},
    {"ranks": ["Q", "J"], "bucket": 1},
]


def bucket_hand(hand: tuple[str, str]) -> int:
    """Return a bucket index for the given hand based on high card."""
    ranks = [card[0] for card in hand]
    for mapping in HAND_BUCKETS:
        if any(r in ranks for r in mapping["ranks"]):
            return mapping["bucket"]
    return len(HAND_BUCKETS)


def abstract_action(action: str) -> str:
    """Map concrete actions to abstracted buckets."""
    action = action.lower()
    if action in {"bet", "raise"}:
        return "aggressive"
    if action in {"check", "call"}:
        return "passive"
    return action
