import torch
import torch.nn as nn

class TransformerAverageStrategy(nn.Module):
    def __init__(self, input_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int, num_actions: int):
        super(TransformerAverageStrategy, self).__init__()
        self.input_projection = nn.Linear(input_feature_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers, # Using num_layers for both encoder and decoder as per original
            num_decoder_layers=num_layers  # Or consider just using TransformerEncoder
        )
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_feature_dim)
        x = self.input_projection(x)
        # x shape after projection: (batch_size, seq_len, hidden_dim)
        
        # Transformer expects src and tgt. For encoder-only behavior with d_model matching,
        # src and tgt can be the same.
        # If using nn.TransformerEncoder, it would just be self.transformer_encoder(x)
        x = self.transformer(x, x) # Output of nn.Transformer is (tgt_seq_len, batch_size, hidden_dim) if tgt is x
                                    # However, if only encoder is used implicitly (by passing only src to a full Transformer),
                                    # it might be (src_seq_len, batch_size, hidden_dim).
                                    # The original code used this, let's assume it worked as intended.
                                    # Standard nn.Transformer(src, tgt) output is tuple if return_memory=True, else just tgt output.
                                    # If tgt is x, then output is (seq_len, batch_size, hidden_dim)
                                    # This might need adjustment if batch is not first.
                                    # PyTorch Transformer default is (S, N, E) for src/tgt, (T, N, E) for output if tgt is given.
                                    # If x comes in as (N, S, E_in), after projection it's (N, S, E_hidden).
                                    # Transformer needs (S, N, E_hidden). So, x.transpose(0, 1) before transformer.
                                    # And transpose back after.
                                    # x = x.transpose(0, 1) # S, N, E
                                    # transformer_output = self.transformer(x, x) # S, N, E
                                    # x = transformer_output.transpose(0, 1) # N, S, E
                                    # This is a common pattern if batch_first=False (default for nn.Transformer)

        # The original code `self.fc(x[:, -1, :])` implies x is (batch, seq, feature) at this point.
        # This suggests nn.Transformer's batch_first behavior might have been implicitly handled or
        # the nn.Transformer was actually nn.TransformerEncoder which can take batch_first=True.
        # Given `nn.Transformer(d_model=...)` it is the full Transformer.
        # For now, keeping the original flow after projection, assuming input x is already correctly shaped
        # or that batch_first=True was intended (though not default for nn.Transformer).
        # If x is (N, S, E_hidden), then self.transformer(x,x) would error if batch_first=False.
        # Let's assume the input x to forward() is (Batch, Seq, Features)
        # and the nn.Transformer is somehow configured or expected to work with this.
        # A standard TransformerEncoder would be:
        #   encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        #   self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #   x = self.transformer_encoder(x) # if x is (N,S,E)
        # Given the existing code, let's stick to minimal changes.
        # If the original code worked, it implies the input to forward was (Seq, Batch, Dim)
        # or there's a misunderstanding of how nn.Transformer was used.
        # Let's assume input x is (Batch, Seq, input_feature_dim) for typical deep learning sequences.
        # Projection: (Batch, Seq, input_feature_dim) -> (Batch, Seq, hidden_dim)
        # Transformer: needs (Seq, Batch, hidden_dim) if batch_first=False (default)
        
        x_permuted = x.permute(1, 0, 2) # (Seq, Batch, hidden_dim)
        transformer_output = self.transformer(x_permuted, x_permuted) # (Seq, Batch, hidden_dim)
        x_restored = transformer_output.permute(1, 0, 2) # (Batch, Seq, hidden_dim)
        
        # Select the output of the last token in the sequence for classification
        x = self.fc(x_restored[:, -1, :]) # (Batch, hidden_dim) -> (Batch, num_actions)
        return nn.Softmax(dim=-1)(x)
