import unittest
from unittest.mock import patch, MagicMock, call
import torch
import sys
import os
import torch.nn.functional as F # For loss verification

# Adjust the Python path to include the root directory of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trainers.ai_cfr_trainer import AICFRTrainer
# Mocked TransformerAverageStrategy will be used, so direct import not strictly needed for tests,
# but good for type hinting if that were used more formally.
# from ai_models.transformer import TransformerAverageStrategy

class TestAICFRTrainer(unittest.TestCase):

    def _get_minimal_config(self):
        return {
            'model': {
                'd_raw_feature': 3,
                'hidden_dim': 32, # Must be divisible by num_heads for TransformerEncoderLayer
                'num_heads': 2,   # Must be > 0
                'num_layers': 1,
                'num_actions': 2,
                'learning_rate': 0.001
            },
            'training': {
                'save_model_path': 'mock_model.pth'
            },
            'logging': { # Optional, but good to include
                'log_file': 'test_aicfr_trainer.log'
            }
        }

    @patch('trainers.ai_cfr_trainer.TransformerAverageStrategy')
    @patch('trainers.ai_cfr_trainer.optim.Adam')
    def test_init_successful(self, mock_adam, mock_transformer_model):
        """Test successful initialization of AICFRTrainer."""
        mock_model_instance = MagicMock()
        mock_transformer_model.return_value = mock_model_instance

        config = self._get_minimal_config()
        trainer = AICFRTrainer(trainer_config=config)

        mock_transformer_model.assert_called_once_with(
            input_feature_dim=config['model']['d_raw_feature'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            num_actions=config['model']['num_actions']
        )
        mock_adam.assert_called_once_with(mock_model_instance.parameters(), lr=config['model']['learning_rate'])

        self.assertEqual(trainer.num_actions, config['model']['num_actions'])
        self.assertTrue(torch.equal(trainer.cumulative_regret, torch.zeros(config['model']['num_actions'])))
        self.assertTrue(torch.equal(trainer.cumulative_strategy, torch.zeros(config['model']['num_actions'])))
        self.assertIsNotNone(trainer.logger) # Check logger is initialized

    def test_init_missing_d_raw_feature(self):
        """Test ValueError if d_raw_feature is missing."""
        config = self._get_minimal_config()
        del config['model']['d_raw_feature']
        with self.assertRaisesRegex(ValueError, "Missing 'd_raw_feature' in model config."):
            AICFRTrainer(trainer_config=config)

    @patch('trainers.ai_cfr_trainer.TransformerAverageStrategy')
    @patch('trainers.ai_cfr_trainer.optim.Adam')
    def test_init_default_model_params(self, mock_adam, mock_transformer_model):
        """Test that default model parameters are used if some are missing."""
        config = {
            'model': {
                'd_raw_feature': 3,
                'num_actions': 2,
                # hidden_dim, num_heads, num_layers, learning_rate are missing
            },
            'training': {'save_model_path': 'mock_model.pth'}
        }
        AICFRTrainer(trainer_config=config)

        # Check that TransformerAverageStrategy was called with default values
        # These defaults are defined in AICFRTrainer's __init__
        expected_hidden_dim = 128
        expected_num_heads = 8
        expected_num_layers = 2
        expected_lr = 0.001

        mock_transformer_model.assert_called_once_with(
            input_feature_dim=config['model']['d_raw_feature'],
            hidden_dim=expected_hidden_dim,
            num_heads=expected_num_heads,
            num_layers=expected_num_layers,
            num_actions=config['model']['num_actions']
        )
        # Adam is called with the model's parameters and the (defaulted) learning rate
        # We don't have direct access to mock_model_instance.parameters() here without more setup,
        # so we check that Adam was called, and trust the lr was correctly defaulted by the constructor.
        self.assertTrue(mock_adam.called)
        args, kwargs = mock_adam.call_args
        self.assertEqual(kwargs.get('lr'), expected_lr)


    @patch('trainers.ai_cfr_trainer.F.mse_loss')
    def test_train_method(self, mock_mse_loss):
        """Test the train method's interactions and updates."""
        config = self._get_minimal_config()
        num_actions = config['model']['num_actions']

        # Patch the model and optimizer within the trainer instance for this test
        with patch.object(AICFRTrainer, '__init__', lambda self, trainer_config: None): # Bypass __init__
            trainer = AICFRTrainer(trainer_config={}) # Config not used due to __init__ bypass
            trainer.model = MagicMock()
            trainer.optimizer = MagicMock()
            trainer.logger = MagicMock() # Mock logger to avoid issues if called
            trainer.num_actions = num_actions
            trainer.cumulative_regret = torch.zeros(num_actions)
            trainer.cumulative_strategy = torch.zeros(num_actions)

        # Mock model's forward pass output
        mock_strategy_pred = torch.rand(num_actions)
        mock_strategy_pred /= torch.sum(mock_strategy_pred) # Normalize
        trainer.model.return_value.squeeze.return_value = mock_strategy_pred

        # Dummy inputs for train method
        state_tensor = torch.rand(1, 5, config['model']['d_raw_feature']) # batch_size=1, seq_len=5
        all_counterfactual_payoffs = torch.rand(num_actions)

        # Mock loss value
        mock_loss_value = torch.tensor(0.5, requires_grad=True)
        mock_mse_loss.return_value = mock_loss_value

        # Call train
        trainer.train(state_tensor, all_counterfactual_payoffs)

        # Assertions
        trainer.model.train.assert_called_once()
        trainer.model.assert_called_once_with(state_tensor) # Input should be batched state_tensor

        # Check optimizer calls
        trainer.optimizer.zero_grad.assert_called_once()
        mock_loss_value.backward.assert_called_once() # Check backward on the returned tensor
        trainer.optimizer.step.assert_called_once()

        # Check if cumulative regrets and strategies have changed from zeros
        # Exact values are hard to check without replicating the logic precisely,
        # but they should no longer be all zeros if payoffs lead to non-zero regrets.
        self.assertFalse(torch.all(trainer.cumulative_regret == 0.0).item(),
                         "Cumulative regret should have been updated.")
        self.assertFalse(torch.all(trainer.cumulative_strategy == 0.0).item(),
                         "Cumulative strategy should have been updated.")

        # Check that F.mse_loss was called.
        # The first argument to mse_loss should be strategy_pred (output of model)
        # The second argument is the target policy (current_regret_matched_policy.detach())
        self.assertTrue(mock_mse_loss.called)
        args, _ = mock_mse_loss.call_args
        self.assertTrue(torch.equal(args[0], mock_strategy_pred))
        # args[1] is current_regret_matched_policy, which is calculated internally.
        # We can check its properties: it should be a probability distribution.
        self.assertEqual(args[1].shape, (num_actions,))
        self.assertTrue(torch.all(args[1] >= 0))
        self.assertAlmostEqual(torch.sum(args[1]).item(), 1.0, places=5)


    @patch('torch.save')
    def test_save_model(self, mock_torch_save):
        config = self._get_minimal_config()
        save_path = config['training']['save_model_path']

        with patch.object(AICFRTrainer, '__init__', lambda x, y: None):
            trainer = AICFRTrainer(trainer_config={})
            trainer.config = config # Set config directly
            trainer.model = MagicMock()
            trainer.logger = MagicMock()

        trainer.save_model()
        trainer.model.state_dict.assert_called_once()
        mock_torch_save.assert_called_once_with(trainer.model.state_dict(), save_path)
        trainer.logger.info.assert_any_call(f"Model saved to {save_path}")

    def test_save_model_no_path(self):
        config = self._get_minimal_config()
        del config['training']['save_model_path'] # Remove save path

        with patch.object(AICFRTrainer, '__init__', lambda x, y: None):
            trainer = AICFRTrainer(trainer_config={})
            trainer.config = config
            trainer.model = MagicMock() # Not strictly needed as it should exit before
            trainer.logger = MagicMock()

        trainer.save_model()
        trainer.logger.error.assert_called_once_with("Missing 'save_model_path' in training config. Cannot save model.")


    @patch('torch.load')
    def test_load_model(self, mock_torch_load):
        config = self._get_minimal_config()
        load_path = config['training']['save_model_path']
        mock_state_dict = {'param': torch.tensor([1.0])}
        mock_torch_load.return_value = mock_state_dict

        with patch.object(AICFRTrainer, '__init__', lambda x, y: None):
            trainer = AICFRTrainer(trainer_config={})
            trainer.config = config
            trainer.model = MagicMock()
            trainer.logger = MagicMock()

        trainer.load_model()
        mock_torch_load.assert_called_once_with(load_path)
        trainer.model.load_state_dict.assert_called_once_with(mock_state_dict)
        trainer.model.eval.assert_called_once()
        trainer.logger.info.assert_any_call(f"Model loaded from {load_path}")

    def test_load_model_no_path(self):
        config = self._get_minimal_config()
        del config['training']['save_model_path'] # Remove load path

        with patch.object(AICFRTrainer, '__init__', lambda x, y: None):
            trainer = AICFRTrainer(trainer_config={})
            trainer.config = config
            trainer.model = MagicMock()
            trainer.logger = MagicMock()

        trainer.load_model()
        trainer.logger.error.assert_called_once_with("Missing 'save_model_path' in training config. Cannot load model.")


    def test_get_final_average_strategy_non_zero(self):
        num_actions = 3
        with patch.object(AICFRTrainer, '__init__', lambda x, y: None):
            trainer = AICFRTrainer(trainer_config={})
            trainer.num_actions = num_actions
            trainer.cumulative_strategy = torch.tensor([1.0, 2.0, 3.0]) # Sum = 6.0
            trainer.logger = MagicMock() # get_final_average_strategy does not log by default

        avg_strategy = trainer.get_final_average_strategy()
        expected_strategy = torch.tensor([1/6, 2/6, 3/6])
        self.assertTrue(torch.allclose(avg_strategy, expected_strategy))
        self.assertAlmostEqual(torch.sum(avg_strategy).item(), 1.0, places=6)

    def test_get_final_average_strategy_all_zeros(self):
        num_actions = 4
        # Patch logger at the module level if it's a global logger used by the function
        with patch('trainers.ai_cfr_trainer.logging') as mock_logging_module:
            with patch.object(AICFRTrainer, '__init__', lambda x, y: None):
                trainer = AICFRTrainer(trainer_config={})
                trainer.num_actions = num_actions
                trainer.cumulative_strategy = torch.zeros(num_actions)
                # trainer.logger = MagicMock() # Not needed if module logger is patched

            avg_strategy = trainer.get_final_average_strategy()
            expected_strategy = torch.ones(num_actions) / num_actions
            self.assertTrue(torch.allclose(avg_strategy, expected_strategy))
            self.assertAlmostEqual(torch.sum(avg_strategy).item(), 1.0, places=6)
            mock_logging_module.warning.assert_called_once_with("Cumulative strategy is all zeros. Returning uniform strategy.")


if __name__ == '__main__':
    unittest.main()
