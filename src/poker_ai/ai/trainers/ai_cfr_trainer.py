import torch
import torch.optim as optim
import logging
import yaml
import os
from poker_ai.ai.models.transformer import AdvantageNetwork
# Assuming rules.cfr is accessible from this path. Adjust if necessary.
# e.g., if 'rules' is a top-level directory: from rules.cfr import ...
# If trainers and rules are siblings under a common root (e.g. 'src'): from ..rules.cfr import ...
from poker_ai.rules.cfr import update_regret, calculate_strategy, update_strategy
import torch.nn.functional as F


# Load configuration - This might fail if config.yaml is not in the expected path during execution
# For robustness, consider passing config path or dictionary.
# For now, keeping as is, assuming it's found relative to where the script/module is run.
try:
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning(
        "config.yaml not found. Using default config values for AICFRTrainer."
    )
    # Define a default config structure if file not found, to allow module loading
    config = {
        'logging': {'log_file': 'aicfr_trainer.log'},
        'model': {'hidden_dim': 128, 'num_actions': 10, 'learning_rate': 0.001, 'd_raw_feature': 18, 'max_seq_len': 256},
        'training': {'save_model_path': 'aicfr_model.pth'}
    }


# Setup logging
log_file_path = config['logging']['log_file']
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='a')

# Expose config for package-level access so tests can override it
import sys
sys.modules[__package__ + '.config'] = config

class AICFRTrainer:
    def __init__(self, device: str | None = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        model_config = config.get('model', {})  # Get model sub-config, or empty dict
        hidden_dim = model_config.get('hidden_dim', 128) # Default if not found
        output_dim = model_config.get('num_actions', 10) # Default if not found
        learning_rate = model_config.get('learning_rate', 0.001) # Default if not found

        # Feature dimensions for history sequence and card set summaries
        d_raw_feature = model_config.get('d_raw_feature', 18)
        d_card_feature = model_config.get('d_card_feature', 17)

        self.model = AdvantageNetwork(
            history_feature_dim=d_raw_feature,
            card_feature_dim=d_card_feature,
            hidden_dim=hidden_dim,
            num_heads=8,  # Could also be in config
            num_layers=2,
            num_actions=output_dim,
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_actions = output_dim  # Ensure this is consistent with model output
        # Expose configuration so callers (e.g. self-play) can retrieve model params
        self.config = config

        # Track regrets and strategies per information set.
        # Keys are information set identifiers supplied during training.
        self.cumulative_regret: dict[str, torch.Tensor] = {}
        self.cumulative_strategy: dict[str, torch.Tensor] = {}

    def get_advantages(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the advantages for the given state representation."""

        if hole_summary.ndim == 1:
            hole_summary = hole_summary.unsqueeze(0)
        if community_summary.ndim == 1:
            community_summary = community_summary.unsqueeze(0)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)

        hole_summary = hole_summary.to(self.device)
        community_summary = community_summary.to(self.device)
        history_tensor = history_tensor.to(self.device)
        mask = mask.to(self.device) if mask is not None else None

        with torch.no_grad():
            advantages = self.model(hole_summary, community_summary, history_tensor, src_mask=None)
        return advantages.squeeze(0)

    def train(
        self,
        info_set_id: str,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_tensor: torch.Tensor,
        all_counterfactual_payoffs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """Train the model for one step based on the provided state."""

        try:
            if hole_summary.ndim == 1:
                hole_summary = hole_summary.unsqueeze(0)
            if community_summary.ndim == 1:
                community_summary = community_summary.unsqueeze(0)
            if history_tensor.ndim == 2:
                history_tensor = history_tensor.unsqueeze(0)

            hole_summary = hole_summary.to(self.device)
            community_summary = community_summary.to(self.device)
            history_tensor = history_tensor.to(self.device)
            all_counterfactual_payoffs = all_counterfactual_payoffs.to(self.device)

            # a. Get model's current strategy prediction
            strategy_pred = self.model(
                hole_summary,
                community_summary,
                history_tensor,
                src_mask=None,
            ).squeeze(0)

            # b. Detach for regret calculation
            current_model_strategy_detached = strategy_pred.detach().clone()

            # c. Calculate state value under current strategy
            state_value = torch.sum(current_model_strategy_detached * all_counterfactual_payoffs)

            # d. Calculate action regrets
            action_regrets = all_counterfactual_payoffs - state_value

            if info_set_id not in self.cumulative_regret:
                self.cumulative_regret[info_set_id] = torch.zeros(self.num_actions, device=self.device)
                self.cumulative_strategy[info_set_id] = torch.zeros(self.num_actions, device=self.device)

            cumulative_regret = self.cumulative_regret[info_set_id]
            cumulative_strategy = self.cumulative_strategy[info_set_id]

            # e. Update cumulative regrets
            cumulative_regret = update_regret(cumulative_regret, action_regrets)

            # f. Current regret-matched policy
            current_regret_matched_policy = calculate_strategy(cumulative_regret, self.num_actions)

            # g. Update cumulative strategy
            cumulative_strategy = update_strategy(
                cumulative_strategy, current_regret_matched_policy.detach()
            )

            self.cumulative_regret[info_set_id] = cumulative_regret
            self.cumulative_strategy[info_set_id] = cumulative_strategy

            # h. Loss: train model output to match regret-matched policy
            loss = F.mse_loss(strategy_pred, current_regret_matched_policy.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logging.info(f"Training step completed. Loss: {loss.item()}")

        except Exception as e:  # pragma: no cover - logging path
            logging.error(f"Error during training: {str(e)}", exc_info=True)
            raise

    def save_model(self, model_path=None):
        # Ensure config path is correct or make it an argument
        try:
            if model_path is None:
                model_path = self.config['training']['save_model_path']
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_path)
            logging.info(f"Model saved to {model_path}")
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


    def get_final_average_strategy(self, info_set_id: str):
        """Return the average strategy for a given information set."""
        cumulative_strategy = self.cumulative_strategy.get(info_set_id)
        if cumulative_strategy is None:
            logging.warning(
                "Requested average strategy for unknown information set '%s'. Returning uniform.",
                info_set_id,
            )
            return torch.ones(self.num_actions, device=self.device) / self.num_actions

        sum_cumulative_strategy = torch.sum(cumulative_strategy)
        if sum_cumulative_strategy == 0:
            logging.warning(
                "Cumulative strategy is all zeros for information set '%s'. Returning uniform strategy.",
                info_set_id,
            )
            return torch.ones(self.num_actions, device=self.device) / self.num_actions
        return cumulative_strategy / sum_cumulative_strategy

