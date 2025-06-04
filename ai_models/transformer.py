import torch
import torch.nn as nn

class TransformerAverageStrategy(nn.Module):
    def __init__(self, input_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int, num_actions: int):
        super(TransformerAverageStrategy, self).__init__()
        self.input_projection = nn.Linear(input_feature_dim, hidden_dim)

        # Refactored to use nn.TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,  # This is crucial for (batch, seq, feature) input
            dim_feedforward=hidden_dim * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, src_key_padding_mask=None): # Renamed src_mask to src_key_padding_mask
        # x shape: (batch_size, seq_len, input_feature_dim)
        x = self.input_projection(x)
        # x shape after projection: (batch_size, seq_len, hidden_dim)
        
        # Input x is already (batch_size, seq_len, hidden_dim)
        # nn.TransformerEncoder with batch_first=True expects this shape.
        # Pass the mask as src_key_padding_mask
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # transformer_output shape: (batch_size, seq_len, hidden_dim)
        
        # Select the output of the last token in the sequence for classification
        # This remains the same as transformer_output is already (batch_size, seq_len, hidden_dim)
        x = self.fc(transformer_output[:, -1, :]) # (Batch, hidden_dim) -> (Batch, num_actions)
        return nn.Softmax(dim=-1)(x)
