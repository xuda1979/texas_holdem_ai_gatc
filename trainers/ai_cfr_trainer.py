import torch
import torch.optim as optim
import logging
import yaml
from ai_models.transformer import TransformerAverageStrategy
# Assuming rules.cfr is accessible from this path. Adjust if necessary.
# e.g., if 'rules' is a top-level directory: from rules.cfr import ...
# If trainers and rules are siblings under a common root (e.g. 'src'): from ..rules.cfr import ...
from ..rules.cfr import update_regret, calculate_strategy, update_strategy
import torch.nn.functional as F

# Setup basic logging if no handler is configured yet for the root logger
# This is useful if the module is used standalone without a higher-level logging setup.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AICFRTrainer:
    """
    AICFRTrainer uses a neural network (TransformerAverageStrategy) to learn a policy.
    The training approach is a variant of Deep CFR, specifically policy regression.
    It maintains global cumulative regrets and strategies. In each training step,
    for a given state:
    1. The network predicts a strategy.
    2. Counterfactual payoffs for this state are provided externally.
    3. Action regrets are calculated using the network's (detached) strategy and the payoffs.
    4. These action regrets update the *global* cumulative regret.
    5. A new target policy is derived from this updated global cumulative regret using regret matching.
    6. This target policy is used as the label for training the network's output.
    7. The global cumulative strategy is updated with this target policy.

    This means the network learns to predict a policy that is consistent with the
    average regret-matched policy derived from all states encountered so far.
    The `cumulative_regret` and `cumulative_strategy` are not per information set
    in the traditional CFR sense, but rather global averages. This approach is suited
    for online learning where the model adapts to a sequence of game states.
    """
    def __init__(self, trainer_config: dict):
        """
        Initializes the AICFRTrainer.
        Args:
            trainer_config: A dictionary containing configuration parameters.
                            Expected keys:
                            - 'model': {
                                'd_raw_feature': int,
                                'hidden_dim': int,
                                'num_heads': int, (optional, default 8)
                                'num_layers': int, (optional, default 2)
                                'num_actions': int,
                                'learning_rate': float
                              }
                            - 'training': { 'save_model_path': str }
                            - 'logging': { 'log_file': str } (optional)
        """
        self.config = trainer_config
        
        log_file = self.config.get('logging', {}).get('log_file')
        if log_file:
            # Add file handler if specific log file is provided
            file_handler = logging.FileHandler(log_file, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            # Get logger for this specific class or a general one
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False # Avoid duplicate logs if root logger is also configured
        else:
            self.logger = logging.getLogger(self.__class__.__name__) # Use existing logger

        model_cfg = self.config.get('model', {})
        d_raw_feature = model_cfg.get('d_raw_feature')
        if d_raw_feature is None:
            self.logger.error("Missing 'd_raw_feature' in model config.")
            raise ValueError("Missing 'd_raw_feature' in model config.")
            
        hidden_dim = model_cfg.get('hidden_dim', 128)
        num_heads = model_cfg.get('num_heads', 8) # Added default
        num_layers = model_cfg.get('num_layers', 2) # Added default
        self.num_actions = model_cfg.get('num_actions', 10) # Also store num_actions
        learning_rate = model_cfg.get('learning_rate', 0.001)

        self.model = TransformerAverageStrategy(
            input_feature_dim=d_raw_feature,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=self.num_actions
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Global cumulative regret and strategy tensors. These are updated across
        # different states if `train` is called with varying `state_tensor`s.
        # This implies an "average" behavior over the sequence of states encountered.
        self.cumulative_regret = torch.zeros(self.num_actions)
        self.cumulative_strategy = torch.zeros(self.num_actions)

    def train(self, state_tensor: torch.Tensor, all_counterfactual_payoffs: torch.Tensor):
        """
        Trains the model for one step based on the provided state and counterfactual payoffs.
        This method implements an online learning approach where global cumulative regrets
        and strategies are updated with experiences from individual states.

        Args:
            state_tensor: A tensor representing the game state. Expected shape [seq_len, feature_dim]
                          or [1, seq_len, feature_dim] if already batched.
            all_counterfactual_payoffs: A tensor of counterfactual payoffs for each possible action
                                        from this state. Shape [num_actions]. These are essentially
                                        E[utility | state, action_i taken, then future play].
        """
        try:
            self.model.train() # Ensure model is in training mode (e.g., for dropout, batchnorm)

            # Ensure state_tensor is correctly shaped for the model (batch_size=1)
            if state_tensor.ndim == 2: # Expected [seq_len, feature_dim]
                state_tensor_batched = state_tensor.unsqueeze(0) # Add batch dimension -> [1, seq_len, feature_dim]
            elif state_tensor.ndim == 3 and state_tensor.shape[0] == 1: # Already batched [1, seq_len, feature_dim]
                state_tensor_batched = state_tensor
            else:
                self.logger.error(f"state_tensor has unexpected shape: {state_tensor.shape}. Expected [seq_len, feature_dim] or [1, seq_len, feature_dim].")
                raise ValueError(f"state_tensor has unexpected shape: {state_tensor.shape}")

            # a. Get model's current strategy prediction for the input state.
            # strategy_pred will have shape [num_actions] after squeeze.
            strategy_pred = self.model(state_tensor_batched).squeeze(0)

            # b. Detach strategy_pred for regret calculation.
            # We use the model's current output (policy) to calculate the state value,
            # which then allows us to compute regrets. This part of the computation
            # should not contribute to the gradients of strategy_pred itself.
            current_model_strategy_detached = strategy_pred.detach().clone()

            # c. Calculate state value under the model's current (detached) strategy.
            # This is V(state) = sum_over_actions [ strategy(action) * counterfactual_payoff(action) ]
            # It represents the expected utility if the current model's policy is followed from this state.
            state_value = torch.sum(current_model_strategy_detached * all_counterfactual_payoffs)

            # d. Calculate action regrets for the current state.
            # Regret(action) = CounterfactualPayoff(action) - V(state)
            # These are the immediate regrets for not choosing optimally according to current info for *this* state.
            action_regrets = all_counterfactual_payoffs - state_value

            # e. Update *global* cumulative regrets with the regrets from the current state.
            # self.cumulative_regret is shared and updated across all states encountered by this trainer instance.
            # This means regrets from different states are being aggregated together.
            self.cumulative_regret = update_regret(self.cumulative_regret, action_regrets)

            # f. Calculate the current iteration's regret-matched policy.
            # This policy is derived from the *global* cumulative regrets. It represents the strategy
            # that minimizes long-term regret based on the average experience across all states seen so far.
            current_regret_matched_policy = calculate_strategy(self.cumulative_regret, self.num_actions)

            # g. Update *global* cumulative strategy by accumulating the current regret-matched policy.
            # This cumulative_strategy is used to compute the final average strategy over the entire training process.
            # The policy added here is detached as it's derived from regrets, not directly from the network's trainable parameters.
            self.cumulative_strategy = update_strategy(self.cumulative_strategy, current_regret_matched_policy.detach())
            
            # h. Compute Loss: Train the model's output (strategy_pred) to match the current_regret_matched_policy.
            # The network is trained to predict the policy that is optimal with respect to the
            # *global* cumulative regrets. The target (current_regret_matched_policy) is detached
            # as it's treated as a fixed label for this training step.
            loss = F.mse_loss(strategy_pred, current_regret_matched_policy.detach())

            # i. Optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Use the instance logger
            self.logger.info(f"Training step completed. Loss: {loss.item()}")
        
        except Exception as e:
            # Use the instance logger
            self.logger.error(f"Error during training: {str(e)}", exc_info=True)
            # Re-raise or handle as appropriate for the application
            raise

    def save_model(self):
        """Saves the model state dictionary to the path specified in config."""
        save_path = self.config.get('training', {}).get('save_model_path')
        if not save_path:
            self.logger.error("Missing 'save_model_path' in training config. Cannot save model.")
            return
        try:
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"Model saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving model to {save_path}: {str(e)}", exc_info=True)

    def load_model(self):
        """Loads the model state dictionary from the path specified in config."""
        load_path = self.config.get('training', {}).get('save_model_path')
        if not load_path:
            self.logger.error("Missing 'save_model_path' in training config. Cannot load model.")
            return
        try:
            self.model.load_state_dict(torch.load(load_path))
            self.model.eval() # Set model to evaluation mode
            self.logger.info(f"Model loaded from {load_path}")
        except Exception as e:
            self.logger.error(f"Error loading model from {load_path}: {str(e)}", exc_info=True)

    def get_final_average_strategy(self):
        # Normalize the cumulative strategy to get the average strategy
        # Avoid division by zero if cumulative_strategy is all zeros
        sum_cumulative_strategy = torch.sum(self.cumulative_strategy)
        if sum_cumulative_strategy == 0:
            # Return a uniform strategy if no strategy has been accumulated
            logging.warning("Cumulative strategy is all zeros. Returning uniform strategy.")
            return torch.ones(self.num_actions) / self.num_actions
        return self.cumulative_strategy / sum_cumulative_strategy

```
