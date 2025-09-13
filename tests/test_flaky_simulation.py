import random

from poker_ai.utils.seeding import set_seed


def draw_card() -> int:
    """Return a pseudo-random card index between 0 and 51."""
    return int(random.random() * 52)


def test_draw_card_deterministic() -> None:
    """Ensure seeding fixes previously flaky randomness."""
    set_seed(42)
    assert draw_card() == 33
