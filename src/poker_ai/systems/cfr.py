"""Subsystem wrapper around the CFR trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer
from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer
from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer

from .base import Subsystem


@dataclass
class CFRSubsystem(Subsystem[AICFRTrainer]):
    """Expose a uniform interface around CFR trainer implementations."""

    @classmethod
    def create(cls, *, variant: str = "ai", **trainer_kwargs: Any) -> "CFRSubsystem":
        variant = variant.lower()
        if variant == "ai":
            trainer = AICFRTrainer(**trainer_kwargs)
        elif variant == "deep":
            trainer = DeepCFRTrainer(**trainer_kwargs)
        elif variant in {"single", "single_network"}:
            trainer = SingleNetworkCFRTrainer(**trainer_kwargs)
        else:  # pragma: no cover - defensive programming
            raise ValueError(f"Unsupported CFR variant: {variant}")
        subsystem = cls(name=f"cfr:{variant}", component=trainer)
        return subsystem

    def train_from_buffer(self, batch_size: int = 256) -> Any:
        """Run one optimisation step using the trainer's replay buffer."""

        return self.component.train(batch_size=batch_size)

    def evaluate_state(self, *args: Any, **kwargs: Any):
        """Proxy through to the trainer's advantage computation."""

        return self.component.get_advantages(*args, **kwargs)
