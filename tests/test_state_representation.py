import os
import sys

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
    _encode_card,
    prepare_transformer_input,
)


def test_prepare_transformer_input_mask() -> None:
    p0 = MockPlayer("p0", ["Ah", "Ks"], 100)
    gs = MockGameState([p0], [], 0, 0, "pre-flop", [], ["p0"])
    encoder = CardSetTransformer(card_dim=17, hidden_dim=32)
    hole, community, seq, mask = prepare_transformer_input(
        gs, "p0", 5, 3, set_encoder=encoder, return_mask=True
    )
    assert hole.shape == (encoder.output_dim,)
    assert community.shape == (encoder.output_dim,)
    assert seq.shape == (5, 3)
    assert mask.shape == (5,)


def test_prepare_transformer_input_engine_game() -> None:
    game = TexasHoldem(num_players=2, starting_stack=100)
    game.initialize_game()
    hole, community, seq = prepare_transformer_input(game, 0, 5, 3)
    assert hole.shape[-1] == community.shape[-1]
    assert seq.shape == (5, 3)


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
