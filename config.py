"""Convenience imports for configuration constants.

This module re-exports key paths used across the project without relying on
wildcard imports, which can obscure the module's public interface and trigger
lint errors.
"""

from poker_ai.config import BASE_DATA_DIR, MODEL_DIR, SIMULATED_DATA_DIR

__all__ = ["BASE_DATA_DIR", "MODEL_DIR", "SIMULATED_DATA_DIR"]
