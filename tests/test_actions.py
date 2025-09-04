import os
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (project_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from gatc_holdem.core.actions import Action, ActionType  # noqa: E402


def test_action_requires_amount_for_bet_like() -> None:
    with pytest.raises(ValueError):
        Action(ActionType.BET)
    with pytest.raises(ValueError):
        Action(ActionType.RAISE, amount_to=-5)
    with pytest.raises(ValueError):
        Action(ActionType.ALL_IN, amount_to=-1)


def test_check_and_call_do_not_need_amount() -> None:
    Action(ActionType.CHECK)
    Action(ActionType.CALL)
