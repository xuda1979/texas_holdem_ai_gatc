"""GUI package with lazy imports.

Importing :mod:`poker_ai.gui` previously pulled in the full Tk/Pillow GUI stack
via :mod:`poker_ai.gui.gui` even when only the strategy helpers were required.
This caused ``ModuleNotFoundError`` for optional dependencies like Pillow in
non-GUI environments (e.g. running CLI games).  We expose the strategy classes
directly and provide a thin wrapper for ``PokerGameGUI`` that imports the heavy
module only on demand.
"""

from .playStrategy import (
    HumanStrategy,
    ModelAIStrategy,
    PlayerStrategy,
    RandomAIStrategy,
)


def PokerGameGUI(*args, **kwargs):  # pragma: no cover - simple wrapper
    from .gui import PokerGameGUI as _PokerGameGUI

    return _PokerGameGUI(*args, **kwargs)


__all__ = [
    "PokerGameGUI",
    "PlayerStrategy",
    "RandomAIStrategy",
    "HumanStrategy",
    "ModelAIStrategy",
]
