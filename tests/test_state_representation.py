import os
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.texas_holdem import TexasHoldem  # noqa: E402
from poker_ai.utils.state_representation import (  # noqa: E402
    RANK_TO_NUM,
    SUIT_TO_NUM,
    CardSetTransformer,
    MockGameState,
    MockPlayer,
    TYPE_ID_CURRENT_BET,
    TYPE_ID_PLAYER_STACK,
    TYPE_ID_POT,
    _encode_card,
    prepare_transformer_input,
)


def test_prepare_transformer_input_mask() -> None:
    p0 = MockPlayer("p0", ["Ah", "Ks"], 100)
    gs = MockGameState([p0], [], 0, 0, "pre-flop", [], ["p0"])
    encoder = CardSetTransformer(card_dim=17, hidden_dim=32)
    hole, community, seq, mask = prepare_transformer_input(
        gs,
        "p0",
        5,
        3,
        normalization_scale=100.0,
        set_encoder=encoder,
        return_mask=True,
    )
    assert hole.shape == (encoder.output_dim,)
    assert community.shape == (encoder.output_dim,)
    assert seq.shape == (5, 3)
    assert mask.shape == (5,)


def test_prepare_transformer_input_engine_game() -> None:
    game = TexasHoldem(num_players=2, starting_stack=100)
    game.initialize_game()
    hole, community, seq = prepare_transformer_input(
        game,
        0,
        5,
        3,
        normalization_scale=float(game.rules.starting_stack),
    )
    assert hole.shape[-1] == community.shape[-1]
    assert seq.shape == (5, 3)
def test_prepare_transformer_input_normalization_default_stack() -> None:
    p0 = MockPlayer("p0", ["Ah", "Ks"], 10_000)
    p1 = MockPlayer("p1", ["Qh", "Js"], 10_000)
    history = [("p0", ("bet", 500))]
    gs = MockGameState(
        [p0, p1],
        ["2c", "3d", "4h"],
        pot=1_500,
        current_bet=500,
        betting_round="flop",
        betting_history=history,
        player_order=["p0", "p1"],
    )
    _, _, seq = prepare_transformer_input(
        gs,
        "p0",
        6,
        3,
        normalization_scale=10_000.0,
    )
    rows = seq.tolist()
    assert rows[0][2] == pytest.approx(0.05)  # 500 / 10000
    history_len = len(history)
    pot_row = rows[history_len]
    assert pot_row[1] == TYPE_ID_POT
    assert pot_row[0] == pytest.approx(0.15)
    bet_row = rows[history_len + 1]
    assert bet_row[1] == TYPE_ID_CURRENT_BET
    assert bet_row[0] == pytest.approx(0.05)
    stack_row = rows[history_len + 2]
    assert stack_row[1] == TYPE_ID_PLAYER_STACK
    assert stack_row[0] == pytest.approx(1.0)


def test_prepare_transformer_input_normalization_small_stack() -> None:
    p0 = MockPlayer("p0", ["Ah", "Ks"], 200)
    p1 = MockPlayer("p1", ["Qh", "Js"], 200)
    p0.current_bet_in_round = 20
    history = [("p1", ("bet", 20)), ("p0", ("call", 20))]
    gs = MockGameState(
        [p0, p1],
        ["2c", "3d", "4h"],
        pot=40,
        current_bet=20,
        betting_round="pre-flop",
        betting_history=history,
        player_order=["p0", "p1"],
    )
    _, _, seq = prepare_transformer_input(
        gs,
        "p0",
        6,
        3,
        normalization_scale=200.0,
    )
    rows = seq.tolist()
    assert rows[0][2] == pytest.approx(0.1)  # 20 / 200
    assert rows[1][2] == pytest.approx(0.1)
    history_len = len(history)
    pot_row = rows[history_len]
    assert pot_row[1] == TYPE_ID_POT
    assert pot_row[0] == pytest.approx(0.2)
    bet_row = rows[history_len + 1]
    assert bet_row[1] == TYPE_ID_CURRENT_BET
    assert bet_row[0] == pytest.approx(0.0)
    stack_row = rows[history_len + 2]
    assert stack_row[1] == TYPE_ID_PLAYER_STACK
    assert stack_row[0] == pytest.approx(1.0)


def test_encode_card_full_deck() -> None:
    ranks = list(RANK_TO_NUM.keys())
    suits = list(SUIT_TO_NUM.keys())
    for i, r in enumerate(ranks):
        for j, s in enumerate(suits):
            vec = _encode_card(r + s)
            assert len(vec) == 17
            assert vec[i] == 1.0
            assert vec[13 + j] == 1.0
            # Only two ones in vector
            assert sum(vec) == 2.0
