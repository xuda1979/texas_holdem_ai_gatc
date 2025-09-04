import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (project_root, src_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.game_state import GameState  # noqa: E402


class DummyPlayer:
    def __init__(self, player_id: int) -> None:
        self.player_id = player_id


def test_game_state_operations() -> None:
    state = GameState()
    players = [DummyPlayer(i) for i in range(3)]
    state.set_players(players)
    assert state.player_order == ["0", "1", "2"]
    assert state.get_player(1) is players[1]
    state.set_player_hand(1, ["As", "Kd"])
    assert state.player_hands[1] == ["As", "Kd"]
    state.add_community_cards(["Qh", "Jh"])
    assert state.community_cards == ["Qh", "Jh"]
    state.record_action(1, ("bet", 100))
    assert state.betting_history[-1] == (1, ("bet", 100))
    state.set_current_bet(150)
    assert state.current_bet == 150
