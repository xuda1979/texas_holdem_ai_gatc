import unittest
from unittest.mock import patch, MagicMock
import torch
from poker_ai.cli.train import initialize_trainer

class TestMultiNPU(unittest.TestCase):

    def test_multi_npu_wrapping(self):
        """
        Tests that the trainer's model is wrapped in DataParallel for multi-NPU training.
        """
        # Mock torch.npu since it doesn't exist in the test environment
        mock_npu = MagicMock()
        mock_npu.is_available.return_value = True
        mock_npu.device_count.return_value = 4

        # Mock torch.zeros to ignore the 'device' kwarg for 'npu'
        original_zeros = torch.zeros
        def mock_zeros(*args, **kwargs):
            if 'device' in kwargs and kwargs['device'] == 'npu':
                kwargs['device'] = 'cpu'
            return original_zeros(*args, **kwargs)

        # Mock the .to() method of nn.Module to avoid the RuntimeError for the 'npu' device
        with patch.object(torch, 'npu', mock_npu, create=True), \
             patch('torch.nn.Module.to', lambda self, *args, **kwargs: self), \
             patch('torch.zeros', mock_zeros):
            algorithms = ["deep_cfr", "ai_cfr", "single_network"]
            dummy_config = {
                "model": {
                    "d_raw_feature": 18,
                    "hidden_dim": 128,
                    "num_actions": 6,
                    "learning_rate": 1e-4
                },
                "logging": {"log_file": "test.log"},
                "training": {"save_model_path": "test_model.pth"}
            }

            for algorithm in algorithms:
                with self.subTest(algorithm=algorithm):
                    trainer = initialize_trainer(
                        algorithm=algorithm,
                        config=dummy_config,
                        device="npu",
                        use_all_npus=True
                    )

                    model_attr = 'advantage_net' if algorithm == 'deep_cfr' else 'model'
                    self.assertTrue(hasattr(trainer, model_attr), f"Trainer for {algorithm} does not have attribute {model_attr}")
                    model = getattr(trainer, model_attr)
                    self.assertIsInstance(model, torch.nn.DataParallel, f"Model for {algorithm} is not wrapped in DataParallel")

if __name__ == '__main__':
    unittest.main()
