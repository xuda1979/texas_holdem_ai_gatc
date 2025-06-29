"""Convenience wrapper for CFR-related functions and classes."""

from rules.cfr import (
    calculate_strategy,
    update_regret,
    update_strategy,
    compute_regrets,
)

# Re-export CFRTrainer so tests can import it from here
from cfr_trainer import CFRTrainer

__all__ = [
    "calculate_strategy",
    "update_regret",
    "update_strategy",
    "compute_regrets",
    "CFRTrainer",
]
