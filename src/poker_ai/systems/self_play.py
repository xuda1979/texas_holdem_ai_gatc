"""Subsystem orchestrating self-play simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_ai.selfplay.self_play import SelfPlay

from .base import Subsystem


@dataclass(slots=True)
class SelfPlaySubsystem(Subsystem[SelfPlay]):
    """Bundle the :class:`SelfPlay` implementation with configuration metadata."""

    @classmethod
    def create(
        cls,
        *,
        cfr_trainer: Any,
        game_engine_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        train_during_generation: bool | None = None,
    ) -> "SelfPlaySubsystem":
        engine_cfg = {
            "starting_stack": 1000,
            "big_blind": 10,
            "small_blind": 5,
            "min_players": 2,
            "max_players": 6,
        }
        if game_engine_config:
            engine_cfg.update(game_engine_config)
        self_play = SelfPlay(
            cfr_trainer=cfr_trainer,
            game_engine_config=engine_cfg,
            training_config=training_config,
            train_during_generation=train_during_generation,
        )
        subsystem = cls(name="self_play", component=self_play)
        subsystem.inject("cfr_trainer", cfr_trainer)
        return subsystem

    def generate_training_hand(self, iteration: int = 0) -> list[Any]:
        return self.component.play_hand_for_training(iteration)
