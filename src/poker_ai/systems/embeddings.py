"""Subsystem responsible for converting game state into model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from poker_ai.utils.state_representation import prepare_transformer_input

from .base import Subsystem


@dataclass
class EmbeddingPipeline:
    """Utility object that bundles the state representation helpers."""

    history_feature_dim: int
    card_feature_dim: int
    max_seq_len: int

    def build_inputs(
        self,
        game: Any,
        player_index: int,
        *,
        normalization_scale: float | None = None,
        device: str | torch.device | None = None,
        return_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hole, community, history, mask = prepare_transformer_input(
            game,
            player_index,
            self.max_seq_len,
            self.history_feature_dim,
            normalization_scale=normalization_scale,
            return_mask=True,
        )
        if device is not None:
            hole = hole.to(device)
            community = community.to(device)
            history = history.to(device)
            mask = mask.to(device)
        if return_mask:
            return hole, community, history, mask
        return hole, community, history, None


@dataclass
class EmbeddingSubsystem(Subsystem[EmbeddingPipeline]):
    """Thin wrapper that constructs :class:`EmbeddingPipeline` objects."""

    @classmethod
    def create(
        cls,
        *,
        history_feature_dim: int,
        card_feature_dim: int,
        max_seq_len: int,
    ) -> "EmbeddingSubsystem":
        pipeline = EmbeddingPipeline(
            history_feature_dim=history_feature_dim,
            card_feature_dim=card_feature_dim,
            max_seq_len=max_seq_len,
        )
        return cls(name="embeddings", component=pipeline)
