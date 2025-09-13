import torch
import torch.nn as nn

from ai_models.transformer import TransformerAverageStrategy


class OpponentModel(nn.Module):
    """Simple Transformer encoder predicting opponent action probabilities."""
    def __init__(self, input_feature_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.transformer = TransformerAverageStrategy(
            input_feature_dim=input_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            num_actions=num_actions,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.transformer(state.unsqueeze(0)).squeeze(0)
