import torch
import torch.nn as nn

class TransformerAverageStrategy(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, num_actions):
        super(TransformerAverageStrategy, self).__init__()
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return nn.Softmax(dim=-1)(x)
