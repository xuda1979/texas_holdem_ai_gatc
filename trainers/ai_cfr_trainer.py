import torch
import torch.optim as optim
import logging
import yaml
import os
from ai_models.transformer import TransformerAverageStrategy
# Assuming rules.cfr is accessible from this path. Adjust if necessary.
# e.g., if 'rules' is a top-level directory: from rules.cfr import ...
# If trainers and rules are siblings under a common root (e.g. 'src'): from ..rules.cfr import ...
from rules.cfr import update_regret, calculate_strategy, update_strategy
import torch.nn.functional as F


# Load configuration - This might fail if config.yaml is not in the expected path during execution
# For robustness, consider passing config path or dictionary.
# For now, keeping as is, assuming it's found relative to where the script/module is run.
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Warning: config.yaml not found. Using default config values for AICFRTrainer.")
    # Define a default config structure if file not found, to allow module loading
    config = {
        'logging': {'log_file': 'aicfr_trainer.log'},
        'model': {'hidden_dim': 128, 'num_actions': 10, 'learning_rate': 0.001},
        'training': {'save_model_path': 'aicfr_model.pth'}
    }


# Setup logging
log_file_path = config['logging']['log_file']
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='a')

class AICFRTrainer:
    def __init__(self, device: str | None = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        model_config = config.get('model', {})  # Get model sub-config, or empty dict
        hidden_dim = model_config.get('hidden_dim', 128) # Default if not found
        output_dim = model_config.get('num_actions', 10) # Default if not found
        learning_rate = model_config.get('learning_rate', 0.001) # Default if not found
        
        # Fetch d_raw_feature with a default value
        # This value should match the d_raw_feature used in state_representation.py
        d_raw_feature = model_config.get('d_raw_feature', 3) 

        self.model = TransformerAverageStrategy(
            input_feature_dim=d_raw_feature,
            hidden_dim=hidden_dim,
            num_heads=8,  # Assuming num_heads and num_layers are fixed or could also be in config
            num_layers=2,
            num_actions=output_dim
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_actions = output_dim  # Ensure this is consistent with model output
        # Expose configuration so callers (e.g. self-play) can retrieve model params
        self.config = config

        # Initialize cumulative regret and strategy tensors
        # These should be persistent across training iterations for a given state-space node if traditional CFR.
        # For deep CFR, these are often associated with the overall training process rather than per-state.
        # The prompt implies these are class members, updated per training step on a specific state.
        # This means they represent the regret/strategy for the "average" state encountered, or this trainer
        # is intended for a single info set (not typical for deep CFR).
        # Given the context, these likely track average regrets/strategies over the samples seen by the network.
        self.cumulative_regret = torch.zeros(self.num_actions, device=self.device)
        self.cumulative_strategy = torch.zeros(self.num_actions, device=self.device)


    def train(self, state_tensor: torch.Tensor, all_counterfactual_payoffs: torch.Tensor):
        """
        Trains the model for one step based on the provided state and counterfactual payoffs.
        Args:
            state_tensor: A tensor representing the game state. Expected shape [seq_len, feature_dim].
            all_counterfactual_payoffs: A tensor of payoffs for each possible action from this state. Shape [num_actions].
        """
        try:
            # Ensure state_tensor is correctly shaped for the model (batch_size=1)
            if state_tensor.ndim == 2: # Should be [seq_len, feature_dim]
                state_tensor_batched = state_tensor.unsqueeze(0)
            elif state_tensor.ndim == 3 and state_tensor.shape[0] == 1: # Already batched
                state_tensor_batched = state_tensor
            else:
                raise ValueError(f"state_tensor has unexpected shape: {state_tensor.shape}")

            state_tensor_batched = state_tensor_batched.to(self.device)
            all_counterfactual_payoffs = all_counterfactual_payoffs.to(self.device)

            # a. Get model's current strategy prediction
            strategy_pred = self.model(state_tensor_batched).squeeze(0)

            # b. Detach strategy_pred for regret calculation
            current_model_strategy_detached = strategy_pred.detach().clone()

            # c. Calculate state value under the model's current (detached) strategy
            state_value = torch.sum(current_model_strategy_detached * all_counterfactual_payoffs)

            # d. Calculate action regrets
            action_regrets = all_counterfactual_payoffs - state_value

            # e. Update cumulative regrets
            # Assuming update_regret handles device placement if necessary, or all tensors are on same device
            self.cumulative_regret = update_regret(self.cumulative_regret, action_regrets)

            # f. Get current iteration's regret-matched policy
            current_regret_matched_policy = calculate_strategy(self.cumulative_regret, self.num_actions)

            # g. Update cumulative strategy (accumulate the regret-matched policy)
            self.cumulative_strategy = update_strategy(self.cumulative_strategy, current_regret_matched_policy.detach())
            
            # h. Compute Loss: Train model's output (strategy_pred) to match current_regret_matched_policy
            # Target should be detached as we don't want to backprop through its calculation.
            loss = F.mse_loss(strategy_pred, current_regret_matched_policy.detach())

            # i. Optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logging.info(f"Training step completed. Loss: {loss.item()}")
        
        except Exception as e:
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            # Re-raise or handle as appropriate for the application
            raise

    def save_model(self):
        # Ensure config path is correct or make it an argument
        try:
            model_path = config['training']['save_model_path']
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            logging.info(f"Model saved to {config['training']['save_model_path']}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}", exc_info=True)


    def load_model(self):
        # Ensure config path is correct or make it an argument
        try:
            self.model.load_state_dict(torch.load(config['training']['save_model_path'], map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded from {config['training']['save_model_path']}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}", exc_info=True)


    def get_final_average_strategy(self):
        # Normalize the cumulative strategy to get the average strategy
        # Avoid division by zero if cumulative_strategy is all zeros
        sum_cumulative_strategy = torch.sum(self.cumulative_strategy)
        if sum_cumulative_strategy == 0:
            # Return a uniform strategy if no strategy has been accumulated
            logging.warning("Cumulative strategy is all zeros. Returning uniform strategy.")
            return torch.ones(self.num_actions) / self.num_actions
        return self.cumulative_strategy / sum_cumulative_strategy

