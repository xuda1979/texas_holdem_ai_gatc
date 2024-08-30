import torch
import torch.optim as optim
import logging
import yaml
from ai_models.transformer import TransformerAverageStrategy
from cfr import calculate_average_strategy, update_cumulative_regret, update_cumulative_strategy

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(filename=config['logging']['log_file'], level=logging.INFO)

class AICFRTrainer:
    def __init__(self):
        input_dim = config['model']['hidden_dim']
        hidden_dim = config['model']['hidden_dim']
        output_dim = config['model']['num_actions']
        
        # Use Transformer model
        self.model = TransformerAverageStrategy(hidden_dim, num_heads=8, num_layers=2, num_actions=output_dim)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['model']['learning_rate'])
        self.cumulative_regret = torch.zeros(output_dim)
        self.cumulative_strategy = torch.zeros(output_dim)
        self.num_actions = output_dim

    def train(self, game_state_sequence, actual_action, actual_payoff):
        try:
            strategy = calculate_average_strategy(self.cumulative_regret, self.num_actions)
            strategy_pred = self.model(game_state_sequence)
            regrets = torch.zeros(self.num_actions)

            for a in range(self.num_actions):
                counterfactual_payoff = self.simulate_payoff_if_action_taken(game_state_sequence, a)
                regrets[a] = counterfactual_payoff - actual_payoff

            self.cumulative_regret = update_cumulative_regret(self.cumulative_regret, regrets)
            self.cumulative_strategy = update_cumulative_strategy(self.cumulative_strategy, strategy)
            
            loss = self.compute_loss(strategy_pred, actual_action, self.cumulative_regret)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logging.info(f"Training step completed. Loss: {loss.item()}")
        
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")

    def compute_loss(self, strategy_pred, actual_action, cumulative_regret):
        return -torch.log(strategy_pred[0][actual_action]) * cumulative_regret[actual_action]

    def simulate_payoff_if_action_taken(self, game_state_sequence, action):
        return torch.rand(1).item()

    def save_model(self):
        torch.save(self.model.state_dict(), config['training']['save_model_path'])

    def load_model(self):
        self.model.load_state_dict(torch.load(config['training']['save_model_path']))

    def get_final_average_strategy(self):
        return self.cumulative_strategy / torch.sum(self.cumulative_strategy)
