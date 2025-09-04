import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (project_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.player import Player  # noqa: E402


def test_player_bet_caps_at_stack() -> None:
    p = Player(player_id=0, stack_size=100)
    assert p.bet(40) == 40
    assert p.stack_size == 60
    assert p.current_bet == 40
    assert p.bet(80) == 60
    assert p.stack_size == 0
    assert p.current_bet == 100
