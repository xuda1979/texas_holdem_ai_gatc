"""Thin wrapper exposing key GUI classes for tests and examples."""

import tkinter as tk
import poker_ai.gui.gui as _gui


class PokerGameGUI(_gui.PokerGameGUI):
    """Subclass that ensures tk reference can be patched in tests."""

    def __init__(self, *args, **kwargs):  # pragma: no cover - simple passthrough
        # allow tests to monkeypatch ``play.gui.tk`` which then flows into the
        # underlying module before initialization
        _gui.tk = tk
        super().__init__(*args, **kwargs)

    # Legacy helper used in tests; reuse base logic but route through patched tk
    def start_game_with_players(self, ai_count: int, starting_stack: int):
        _gui.tk = tk
        total_players = ai_count + 1
        strategies = [GUIHumanStrategy(self)]
        for _ in range(ai_count):
            strategies.append(_gui.RandomAIStrategy())
        self.game = _gui.TexasHoldem(total_players, starting_stack, strategies)
        # These calls are patched out in tests to avoid heavy GUI work
        self.setup_gui()
        self.start_game()


GUIHumanStrategy = _gui.GUIHumanStrategy

__all__ = ["PokerGameGUI", "GUIHumanStrategy", "tk"]
