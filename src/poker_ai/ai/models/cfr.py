"""Convenience wrapper for CFR-related functions and classes."""

# Import the CFR helpers from the package inside ``src``.  Using an absolute
# import avoids accidentally resolving the top-level ``rules`` directory that
# pulls in heavy engine dependencies and requires optional packages such as
# ``treys``.  The helpers themselves are lightweight and independent of the
# game engine, so importing them directly keeps `poker_ai.ai.models` usable in
# environments where the full engine stack is unavailable.
# Re-export CFRTrainer so tests can import it from here
from cfr_trainer import CFRTrainer
from poker_ai.rules.cfr import (
    calculate_strategy,
    compute_regrets,
    update_regret,
    update_strategy,
)

__all__ = [
    "calculate_strategy",
    "update_regret",
    "update_strategy",
    "compute_regrets",
    "CFRTrainer",
]
