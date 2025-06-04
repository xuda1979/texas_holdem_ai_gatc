from utils.state_representation import MockGameState, MockPlayer, prepare_transformer_input

def test_prepare_transformer_input_mask():
    p0 = MockPlayer('p0', ['Ah', 'Ks'], 100)
    gs = MockGameState([p0], [], 0, 0, 'pre-flop', [], ['p0'])
    tensor, mask = prepare_transformer_input(gs, 'p0', 5, 3, return_mask=True)
    assert tensor.shape == (5, 3)
    assert mask.shape == (5,)
