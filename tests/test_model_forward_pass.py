import unittest
import torch
import sys
import os

# Adjust the Python path to include the root directory of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_models.transformer import TransformerAverageStrategy

class TestModelForwardPass(unittest.TestCase):

    def _run_forward_pass_test(self, model_params: dict, input_params: dict, test_description: str):
        """
        Helper function to run a forward pass test with given model and input parameters.
        """
        print(f"\nRunning test: {test_description}")

        # Instantiate the model
        model = TransformerAverageStrategy(
            input_feature_dim=model_params['input_feature_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_heads=model_params['num_heads'],
            num_layers=model_params['num_layers'],
            num_actions=model_params['num_actions']
        )
        model.eval() # Set to evaluation mode
        print(f"  Model instantiated: {model_params}")

        # Create a dummy input tensor
        dummy_input = torch.rand(
            input_params['batch_size'],
            input_params['seq_len'],
            model_params['input_feature_dim']
        )
        print(f"  Dummy input tensor created with shape: {dummy_input.shape}")

        # Perform a forward pass
        with torch.no_grad(): # Disable gradient calculations
            output_probs = model(dummy_input)
        print(f"  Model forward pass successful. Output shape: {output_probs.shape}")

        # Assert the output shape is correct
        expected_output_shape = (input_params['batch_size'], model_params['num_actions'])
        self.assertEqual(output_probs.shape, expected_output_shape,
                         f"Output shape mismatch. Expected {expected_output_shape}, got {output_probs.shape}")

        # Assert output values are probabilities
        self.assertTrue(torch.all(output_probs >= 0) and torch.all(output_probs <= 1),
                        f"Output probabilities are not in [0, 1] range. Values: {output_probs.tolist()}")

        sum_of_probs = torch.sum(output_probs, dim=1)
        expected_sum = torch.ones(input_params['batch_size'])
        self.assertTrue(torch.allclose(sum_of_probs, expected_sum, atol=1e-6),
                        f"Output probabilities do not sum to 1 (within tolerance). Sums: {sum_of_probs.tolist()}")
        print(f"  Assertions PASSED for {test_description}")

    def test_standard_case(self):
        model_params = {
            'input_feature_dim': 3, 'hidden_dim': 128, 'num_heads': 4,
            'num_layers': 2, 'num_actions': 10
        }
        input_params = {'batch_size': 4, 'seq_len': 20}
        self._run_forward_pass_test(model_params, input_params, "Standard Case (batch=4, seq_len=20)")

    def test_batch_size_one(self):
        model_params = {
            'input_feature_dim': 3, 'hidden_dim': 64, 'num_heads': 2,
            'num_layers': 1, 'num_actions': 5
        }
        input_params = {'batch_size': 1, 'seq_len': 10}
        self._run_forward_pass_test(model_params, input_params, "Batch Size One (batch=1, seq_len=10)")

    def test_seq_len_one(self):
        model_params = {
            'input_feature_dim': 5, 'hidden_dim': 32, 'num_heads': 2,
            'num_layers': 1, 'num_actions': 3
        }
        input_params = {'batch_size': 2, 'seq_len': 1}
        self._run_forward_pass_test(model_params, input_params, "Sequence Length One (batch=2, seq_len=1)")

    def test_varied_model_params(self):
        model_params = {
            'input_feature_dim': 10, 'hidden_dim': 256, 'num_heads': 8,
            'num_layers': 4, 'num_actions': 20
        }
        input_params = {'batch_size': 2, 'seq_len': 5}
        self._run_forward_pass_test(model_params, input_params, "Varied Model Parameters")

    def test_minimal_params(self):
        # Smallest reasonable dimensions
        model_params = {
            'input_feature_dim': 1, 'hidden_dim': 16, 'num_heads': 1, # Min heads = 1
            'num_layers': 1, 'num_actions': 2
        }
        # TransformerEncoderLayer requires hidden_dim to be divisible by num_heads.
        # For num_heads=1, any hidden_dim is fine.
        input_params = {'batch_size': 1, 'seq_len': 1}
        self._run_forward_pass_test(model_params, input_params, "Minimal Parameters (batch=1, seq_len=1, feature_dim=1, heads=1)")


if __name__ == '__main__':
    unittest.main()
    # The old runner is replaced by unittest.main()
    # try:
    #     # test_transformer_forward_pass() # This was the old way
    #     # Instead, rely on unittest discovery if run directly, or use `python -m unittest tests.test_model_forward_pass`
    #     print("\nRun tests using 'python -m unittest tests.test_model_forward_pass'")
    # except AssertionError as e:
    #     print(f"\nTest FAILED: {e}")
    # except Exception as e:
    #     print(f"\nAn unexpected error occurred during testing: {e}")
    #     import traceback
    #     traceback.print_exc()
