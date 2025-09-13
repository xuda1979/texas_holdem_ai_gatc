import os
import sys

import pytest

# Ensure project root and src are in path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (SRC_PATH, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.utils.seeding import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def global_seed() -> None:
    """Seed random number generators for every test."""
    seed = int(os.getenv("PYTEST_SEED", "0"))
    set_seed(seed)
