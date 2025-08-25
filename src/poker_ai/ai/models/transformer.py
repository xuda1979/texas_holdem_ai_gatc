import torch
import torch.nn as nn


class AdvantageNetwork(nn.Module):
    """Transformer based advantage network.

    The model expects three inputs:

    * ``hole_summary`` – summary vector of the player's hole cards
    * ``community_summary`` – summary vector of the public cards
    * ``history_seq`` – the sequential history tensor

    The card summaries are projected and concatenated with a pooled
    representation of ``history_seq`` before the final linear layer outputs the
    advantages for each abstract action.
    """

    def __init__(
        self,
        history_feature_dim: int,
        card_feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.history_projection = nn.Linear(history_feature_dim, hidden_dim)
        self.card_projection = nn.Linear(card_feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # We concatenate three hidden vectors (hole, community, history)
        self.fc = nn.Linear(hidden_dim * 3, num_actions)
        self.num_actions = num_actions

    def forward(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_seq: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw advantages for each action."""

        if history_seq.dim() != 3:
            raise ValueError("history_seq should be of shape (batch, seq_len, feat_dim)")

        h = self.history_projection(history_seq)
        h = self.transformer(h, mask=src_mask)
        h = h[:, -1, :]

        hole = self.card_projection(hole_summary)
        community = self.card_projection(community_summary)

        fused = torch.cat([hole, community, h], dim=-1)
        return self.fc(fused)
