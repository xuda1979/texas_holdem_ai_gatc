"""Lightweight strategy stubs for tests and examples."""

from poker_ai.gui.playStrategy import PlayerStrategy


class PlaceholderAIStrategy(PlayerStrategy):
    """Deterministic strategy used only for testing imports."""

    @property
    def is_human(self):
        return False

    def choose_action(self, game, player_index):
        """Always fold to keep behaviour predictable."""
        return "fold", None


__all__ = ["PlaceholderAIStrategy"]
