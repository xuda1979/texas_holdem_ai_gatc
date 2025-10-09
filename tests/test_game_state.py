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


def test_game_state_reset_and_serialization() -> None:
    state = GameState()
    players = [DummyPlayer(i) for i in range(2)]
    state.set_players(players)
    state.set_player_hand(0, ("As", "Kd"))
    state.add_community_cards(["2h", "3d", "4s"])
    state.record_action(0, ("bet", 50))
    state.set_current_bet(50)

    serialized = state.to_dict()
    assert serialized["pot"] == 0
    assert serialized["player_hands"] == {"0": ["As", "Kd"]}
    assert serialized["betting_history"] == [
        {"player_id": "0", "action": "bet", "amount": 50}
    ]
    assert serialized["player_order"] == ["0", "1"]

    round_trip = GameState.from_dict(serialized)
    assert round_trip.current_bet == 50
    assert round_trip.betting_history == [("0", ("bet", 50))]
    assert round_trip.player_hands["0"] == ("As", "Kd")

    state.reset(keep_players=True)
    assert state.players == players
    assert state.player_order == ["0", "1"]
    assert state.community_cards == []
    assert state.player_hands == {}
    assert state.current_bet == 0

    state.reset(keep_players=False)
    assert state.players == []
    assert state.player_order == []
