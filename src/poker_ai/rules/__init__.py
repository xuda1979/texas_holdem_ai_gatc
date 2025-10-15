"""Lightweight export of CFR helpers.

Importing the full Texas Hold'em rule set used to pull in the entire game
engine and its optional dependencies.  Many modules only need the basic CFR
utilities, so we expose those here without eagerly importing the engine
components.  For the complete rules implementation use
``poker_ai.rules.texas_holdem``.
"""

from .cfr import cfr_plus_iteration, update_regret, update_strategy

__all__ = [
    "cfr_plus_iteration",
    "update_regret",
    "update_strategy",
]
