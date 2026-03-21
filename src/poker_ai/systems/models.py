"""Subsystem exposing neural network model factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from poker_ai.ai.models.transformer import AdvantageNetwork

from .base import Subsystem


@dataclass
class TransformerSubsystem(Subsystem[AdvantageNetwork]):
    """Wrap the transformer advantage network used by CFR trainers."""

    @classmethod
    def create(
        cls,
        *,
        history_feature_dim: int,
        card_feature_dim: int,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        num_layers: int | None = None,
        num_actions: int | None = None,
    ) -> "TransformerSubsystem":
        hidden_dim = hidden_dim or AdvantageNetwork.DEFAULT_HIDDEN_DIM
        num_heads = num_heads or AdvantageNetwork.recommended_num_heads(hidden_dim)
        num_layers = num_layers or AdvantageNetwork.DEFAULT_NUM_LAYERS
        num_actions = num_actions or 10
        network = AdvantageNetwork(
            history_feature_dim=history_feature_dim,
            card_feature_dim=card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=num_actions,
        )
        subsystem = cls(
            name="model:transformer",
            component=network,
        )
        return subsystem

    def freeze(self) -> None:
        """Convenience helper for evaluation subsystems."""

        for parameter in self.component.parameters():
            parameter.requires_grad_(False)
        self.component.eval()

    @property
    def parameters(self) -> nn.ParameterList:  # pragma: no cover - trivial proxy
        return self.component.parameters()
