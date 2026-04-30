from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplicitTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer without the fused PyTorch fast path.

    Ascend falls back to CPU for ``torch._transformer_encoder_layer_fwd``.
    Implementing the layer explicitly keeps execution on-device while retaining
    the same parameter structure as ``nn.TransformerEncoderLayer`` so legacy
    checkpoints stay loadable.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = True,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.norm_first = norm_first

    def _sa_block(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(attn_output)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout2(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del is_causal
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class ExplicitTransformerEncoder(nn.Module):
    """Minimal encoder stack mirroring ``nn.TransformerEncoder`` behavior."""

    def __init__(self, encoder_layer: ExplicitTransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [encoder_layer if idx == 0 else type(encoder_layer)(
                d_model=encoder_layer.self_attn.embed_dim,
                nhead=encoder_layer.self_attn.num_heads,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout.p,
                batch_first=encoder_layer.self_attn.batch_first,
                norm_first=encoder_layer.norm_first,
            ) for idx in range(num_layers)]
        )

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class AdvantageNetwork(nn.Module):
    """Transformer based advantage network.

    The model expects three inputs:

    * ``hole_summary`` – summary vector of the player's hole cards
    * ``community_summary`` – summary vector of the public cards
    * ``history_seq`` – the sequential history tensor

    The card summaries are projected and concatenated with a pooled
    representation of ``history_seq`` before the final linear layer outputs the
    advantages for each abstract action.

    ``DEFAULT_NUM_LAYERS`` exposes the project's canonical transformer depth so
    that other modules can adopt the same default without duplicating literals.
    Updating the default depth in this module automatically informs all callers
    that reference the attribute.
    """

    DEFAULT_NUM_LAYERS = 12
    DEFAULT_HIDDEN_DIM = 768
    DEFAULT_NUM_HEADS = 12

    @staticmethod
    def recommended_num_heads(
        hidden_dim: int, preferred: int | None = None
    ) -> int:
        """Return a head count that evenly divides ``hidden_dim``.

        When loading legacy checkpoints that do not record the original head
        count, callers can request the highest divisor up to ``preferred``.
        ``preferred`` defaults to :data:`DEFAULT_NUM_HEADS`, ensuring the
        transformer remains compatible with historical smaller models while
        using twelve heads for the new XL configuration.
        """

        target = preferred or AdvantageNetwork.DEFAULT_NUM_HEADS
        if hidden_dim <= 0:
            return 1
        target = min(target, hidden_dim)
        for candidate in range(target, 0, -1):
            if hidden_dim % candidate == 0:
                return candidate
        return 1

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
        encoder_layer = ExplicitTransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = ExplicitTransformerEncoder(encoder_layer, num_layers=num_layers)
        # We concatenate three hidden vectors (hole, community, history)
        self.fc = nn.Linear(hidden_dim * 3, num_actions)
        self.num_actions = num_actions

    def forward(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_seq: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        """Return raw advantages for each action.

        Parameters
        ----------
        padding_mask:
            Optional boolean mask with shape ``(batch, seq_len)`` where ``True``
            entries indicate positions that should be ignored by the transformer
            encoder (PyTorch's ``src_key_padding_mask`` semantics).
        """

        if key_padding_mask is not None:
            padding_mask = key_padding_mask if padding_mask is None else padding_mask

        if history_seq.dim() != 3:
            raise ValueError("history_seq should be of shape (batch, seq_len, feat_dim)")

        h = self.history_projection(history_seq)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        h = h[:, -1, :]

        hole = self.card_projection(hole_summary)
        community = self.card_projection(community_summary)

        fused = torch.cat([hole, community, h], dim=-1)
        return self.fc(fused)
