import os
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (project_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from gatc_poker.handeval import evaluate_hand  # noqa: E402


def test_invalid_card_string() -> None:
    with pytest.raises(ValueError):
        evaluate_hand(["1x", "Ac"], ["Ad", "Kd", "Qd"])


def test_duplicate_cards() -> None:
    with pytest.raises(ValueError):
        evaluate_hand(["As", "Ac"], ["Ad", "Ks", "As"])


def test_min_and_max_cards() -> None:
    assert isinstance(evaluate_hand(["As", "Kd"], ["2c", "3d", "4h"]), int)
    assert isinstance(evaluate_hand(["As", "Kd"], ["2c", "3d", "4h", "5s", "6c"]), int)
