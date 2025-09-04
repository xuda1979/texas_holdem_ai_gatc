import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (project_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from gatc_holdem.engine.state import GameState, PlayerState  # noqa: E402


def test_reset_street_resets_bets() -> None:
    players = [PlayerState(stack=100, committed=50) for _ in range(2)]
    state = GameState(
        big_blind=100,
        dealer_index=0,
        street="flop",
        current_bet_to=200,
        last_raise_size=100,
        last_aggressor=1,
        players=players,
    )
    state.reset_street("turn")
    assert state.street == "turn"
    assert state.current_bet_to == 0
    assert state.last_raise_size == 0
    assert state.last_aggressor is None
    assert all(p.committed == 0 for p in state.players)
