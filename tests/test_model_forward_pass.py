import torch
import sys
import os

# Adjust the Python path to include the root directory of the project
# This allows importing modules from ai_models, utils etc.
# Assuming 'tests' is a directory at the root of the project.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_models.transformer import TransformerAverageStrategy

def test_transformer_forward_pass():
    print("Running TransformerAverageStrategy Forward Pass Test...")

    # Define model parameters (matching example values from config or typical use)
    # These should ideally be sourced from a test config or match the main config.yaml
    # For d_raw_feature, it's what prepare_transformer_input produces.
    # For num_actions, it's what get_action_from_index maps to.
    input_feature_dim = 3  # Example: from config.yaml model.d_raw_feature
    hidden_dim = 128       # Example: from config.yaml model.hidden_dim
    num_actions = 10       # Example: from config.yaml model.num_actions
    
    # Transformer specific parameters (can be varied for tests if needed)
    num_heads = 4          # Example value, can be 8 as in AICFRTrainer
    num_layers = 2         # Example value

    # Input tensor parameters
    max_seq_len = 20       # Example: from AICFRTrainer Mock config or typical sequence length
    batch_size = 4         # Test with a small batch

    # Instantiate the model
    try:
        model = TransformerAverageStrategy(
            input_feature_dim=input_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=num_actions
        )
        model.eval() # Set to evaluation mode for testing (disables dropout if any)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating model: {e}")
        raise

    # Create a dummy input tensor
    # Shape: (batch_size, max_seq_len, input_feature_dim)
    try:
        dummy_input = torch.rand(batch_size, max_seq_len, input_feature_dim)
        print(f"Dummy input tensor created with shape: {dummy_input.shape}")
    except Exception as e:
        print(f"Error creating dummy input tensor: {e}")
        raise

    # Perform a forward pass
    try:
        with torch.no_grad(): # Disable gradient calculations for inference
            output_probs = model(dummy_input)
        print(f"Model forward pass successful. Output shape: {output_probs.shape}")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        raise

    # Assert the output shape is correct
    expected_output_shape = (batch_size, num_actions)
    assert output_probs.shape == expected_output_shape, \
        f"Output shape mismatch. Expected {expected_output_shape}, got {output_probs.shape}"
    print(f"Assertion for output shape PASSED. Expected {expected_output_shape}, got {output_probs.shape}")

    # Assert output values are probabilities
    # 1. All values should be between 0 and 1 (inclusive)
    assert torch.all(output_probs >= 0) and torch.all(output_probs <= 1), \
        f"Output probabilities are not in [0, 1] range. Values: {output_probs.tolist()}"
    print("Assertion for output probabilities >= 0 and <= 1 PASSED.")

    # 2. Each row (strategy for each item in batch) should sum to 1
    sum_of_probs = torch.sum(output_probs, dim=1)
    expected_sum = torch.ones(batch_size) # Expected sum for each item in batch is 1.0
    assert torch.allclose(sum_of_probs, expected_sum, atol=1e-6), \
        f"Output probabilities do not sum to 1 (within tolerance) for each batch item. Sums: {sum_of_probs.tolist()}"
    print("Assertion for sum of probabilities per batch item == 1 PASSED.")

    print("TransformerAverageStrategy forward pass test completed successfully!")

if __name__ == '__main__':
    try:
        test_transformer_forward_pass()
        print("\nAll tests in test_model_forward_pass.py PASSED.")
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
