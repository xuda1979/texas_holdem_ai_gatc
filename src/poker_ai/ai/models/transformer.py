import torch
import torch.nn as nn


class TransformerAverageStrategy(nn.Module):
    """Encoder-only Transformer producing action probabilities."""

    def __init__(
        self,
        input_feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_actions)
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode ``x`` and return a probability distribution over actions."""
        x = self.input_projection(x)
        x = self.transformer(x, mask=src_mask)
        x = self.fc(x[:, -1, :])
        return nn.Softmax(dim=-1)(x)

