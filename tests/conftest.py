"""Pytest configuration for GUI-related tests."""

import os
from pathlib import Path

import pytest

# Directory containing bundled card images
CARD_IMAGES_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "poker_ai" / "gui" / "card_images"
)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip GUI tests when prerequisites are missing."""
    if "gui" in item.keywords:
        # Ensure card images are available
        if not CARD_IMAGES_DIR.exists() or len(list(CARD_IMAGES_DIR.glob("*.png"))) < 52:
            pytest.skip("card images not available")
        # Ensure a display is available (e.g., via Xvfb)
        if not os.environ.get("DISPLAY"):
            pytest.skip("no display available for GUI test")
