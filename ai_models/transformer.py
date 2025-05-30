import torch
import torch.nn as nn

class TransformerAverageStrategy(nn.Module):
    """
    A strategy model that uses a Transformer Encoder to process sequences of features
    and then averages the transformer's output before passing it to a fully connected layer.
    This model assumes input tensors are of shape (batch_size, seq_len, input_feature_dim).
    """
    def __init__(self, input_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int, num_actions: int):
        super(TransformerAverageStrategy, self).__init__()

        # Project input features to the hidden dimension expected by the transformer
        self.input_projection = nn.Linear(input_feature_dim, hidden_dim)

        # Define the Transformer Encoder layer
        # batch_first=True means input/output tensors are (batch_size, seq_len, feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4, # Standard practice: feedforward_dim is 4*hidden_dim
            batch_first=True  # Crucial for handling (batch, seq, feature) inputs
        )

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # Fully connected layer to map transformer output to action space
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_feature_dim)
        Returns:
            Output tensor of shape (batch_size, num_actions) after softmax.
        """
        # x initial shape: (batch_size, seq_len, input_feature_dim)

        # Project input features to hidden_dim
        x = self.input_projection(x)
        # x shape after projection: (batch_size, seq_len, hidden_dim)
        
        # Pass through Transformer Encoder
        # Input to transformer_encoder: (batch_size, seq_len, hidden_dim)
        # Output from transformer_encoder: (batch_size, seq_len, hidden_dim)
        x = self.transformer_encoder(x)

        # Select the output of the last token in the sequence for classification/action selection.
        # This is a common approach, assuming the last token's representation
        # captures the relevant information from the sequence.
        # x[:, -1, :] selects all batches, the last token in the sequence, and all features.
        # Shape becomes: (batch_size, hidden_dim)
        last_token_output = x[:, -1, :]
        
        # Pass the last token's output through the fully connected layer
        # Shape: (batch_size, hidden_dim) -> (batch_size, num_actions)
        x = self.fc(last_token_output)
        
        # Apply Softmax to get action probabilities
        return nn.Softmax(dim=-1)(x)
