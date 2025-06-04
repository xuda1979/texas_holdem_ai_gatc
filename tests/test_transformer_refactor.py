import torch
import torch.nn as nn
import unittest

# Adjust the import path based on your project structure
# Assuming the tests directory is at the same level as 'ai_models' directory or 'ai_models' is in PYTHONPATH
try:
    from ai_models.transformer import TransformerAverageStrategy
except ImportError:
    # Fallback for different project structures, e.g. when tests is a subdir of a package
    # This assumes '..' is the project root and 'ai_models' is a package there.
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ai_models.transformer import TransformerAverageStrategy


class TestTransformerRefactor(unittest.TestCase):

    def test_transformer_average_strategy_forward_pass(self):
        # Model parameters
        input_feature_dim = 10
        hidden_dim = 32
        num_heads = 4 # hidden_dim (32) must be divisible by num_heads (4)
        num_layers = 2
        num_actions = 5

        # Instantiate the model
        model = TransformerAverageStrategy(
            input_feature_dim=input_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=num_actions
        )

        # Create dummy input tensor
        batch_size = 3
        seq_len = 7
        dummy_input = torch.rand(batch_size, seq_len, input_feature_dim)

        # Perform a forward pass
        output = model(dummy_input)

        # Assert output shape
        expected_output_shape = (batch_size, num_actions)
        self.assertEqual(output.shape, expected_output_shape,
                         f"Output shape mismatch. Expected {expected_output_shape}, got {output.shape}")

        # Assert that output probabilities sum to 1 for each item in the batch
        sum_of_probabilities = torch.sum(output, dim=-1)
        expected_sums = torch.ones(batch_size)
        self.assertTrue(torch.allclose(sum_of_probabilities, expected_sums, atol=1e-6),
                        f"Output probabilities do not sum to 1. Sums: {sum_of_probabilities}")

    def test_transformer_average_strategy_with_mask(self):
        # Model parameters
        input_feature_dim = 10
        hidden_dim = 32
        num_heads = 4
        num_layers = 2
        num_actions = 5

        model = TransformerAverageStrategy(
            input_feature_dim=input_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=num_actions
        )

        batch_size = 2
        seq_len = 4
        dummy_input = torch.rand(batch_size, seq_len, input_feature_dim)

        # Create a source mask (example: mask out the last two elements for the first batch item,
        # and the last element for the second batch item)
        # src_mask for TransformerEncoder should be (N, S) or (S, S).
        # (N,S) mask: True values are positions that will NOT be allowed to attend.
        src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        src_key_padding_mask[0, -2:] = True
        src_key_padding_mask[1, -1:] = True

        # Perform a forward pass with the mask
        # Note: nn.TransformerEncoder expects a src_key_padding_mask
        output = model(dummy_input, src_key_padding_mask=src_key_padding_mask) # Changed src_mask to src_key_padding_mask

        # Assert output shape
        expected_output_shape = (batch_size, num_actions)
        self.assertEqual(output.shape, expected_output_shape,
                         f"Output shape mismatch with mask. Expected {expected_output_shape}, got {output.shape}")

        # Assert that output probabilities sum to 1
        sum_of_probabilities = torch.sum(output, dim=-1)
        expected_sums = torch.ones(batch_size)
        self.assertTrue(torch.allclose(sum_of_probabilities, expected_sums, atol=1e-6),
                        f"Output probabilities do not sum to 1 with mask. Sums: {sum_of_probabilities}")

if __name__ == '__main__':
    unittest.main()
