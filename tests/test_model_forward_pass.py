import os
import sys

import pytest
import torch

# Adjust the Python path to include the root directory of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.ai.models.transformer import AdvantageNetwork


def test_advantage_network_forward_pass():
    print("Running AdvantageNetwork Forward Pass Test...")

    # Define model parameters
    history_feature_dim = 3
    card_feature_dim = 32
    hidden_dim = 128
    num_actions = 10
    num_heads = 4
    num_layers = AdvantageNetwork.DEFAULT_NUM_LAYERS

    # Input tensor parameters
    max_seq_len = 20
    batch_size = 4

    # Instantiate the model
    try:
        model = AdvantageNetwork(
            history_feature_dim=history_feature_dim,
            card_feature_dim=card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=num_actions,
        )
        model.eval()
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating model: {e}")
        raise

    # Create a dummy input tensor
    try:
        dummy_history = torch.rand(batch_size, max_seq_len, history_feature_dim)
        dummy_hole = torch.rand(batch_size, card_feature_dim)
        dummy_community = torch.rand(batch_size, card_feature_dim)
        print(
            f"Dummy tensors created with shapes: history {dummy_history.shape}, hole {dummy_hole.shape}"
        )
    except Exception as e:
        print(f"Error creating dummy input tensor: {e}")
        raise

    # Perform a forward pass
    try:
        with torch.no_grad():
            output_advantages = model(dummy_hole, dummy_community, dummy_history)
        print(f"Model forward pass successful. Output shape: {output_advantages.shape}")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        raise

    # Assert the output shape is correct
    expected_output_shape = (batch_size, num_actions)
    assert (
        output_advantages.shape == expected_output_shape
    ), f"Output shape mismatch. Expected {expected_output_shape}, got {output_advantages.shape}"
    print(
        f"Assertion for output shape PASSED. Expected {expected_output_shape}, got {output_advantages.shape}"
    )

    print("AdvantageNetwork forward pass test completed successfully!")


def test_advantage_network_legacy_transformer_state_dict_compatibility():
    legacy = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True),
        num_layers=2,
    )
    model = AdvantageNetwork(
        history_feature_dim=3,
        card_feature_dim=5,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        num_actions=6,
    )

    transformed_state = {
        f"transformer.{key}": value.clone()
        for key, value in legacy.state_dict().items()
    }

    missing, unexpected = model.load_state_dict(transformed_state, strict=False)

    assert not unexpected
    assert all(not key.startswith("transformer.") for key in missing)
    assert torch.allclose(
        model.transformer.layers[0].self_attn.in_proj_weight,
        legacy.layers[0].self_attn.in_proj_weight,
    )


def test_explicit_transformer_respects_padding_mask():
    torch.manual_seed(0)
    model = AdvantageNetwork(
        history_feature_dim=4,
        card_feature_dim=6,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        num_actions=3,
    )
    model.eval()

    hole = torch.randn(2, 6)
    community = torch.randn(2, 6)
    history = torch.randn(2, 5, 4)
    padding_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, False, True, True, True],
        ]
    )

    with torch.no_grad():
        masked = model(hole, community, history, padding_mask=padding_mask)
        unmasked = model(hole, community, history)

    assert masked.shape == (2, 3)
    assert not torch.allclose(masked, unmasked)


if __name__ == "__main__":
    try:
        test_advantage_network_forward_pass()
        print("\nAll tests in test_model_forward_pass.py PASSED.")
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
        import traceback

        traceback.print_exc()
