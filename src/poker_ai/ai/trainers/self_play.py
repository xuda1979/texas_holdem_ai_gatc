"""Legacy self-play stub.

This module is preserved for backwards-compatibility but is not used by
the modern training pipeline, which lives in
:mod:`poker_ai.selfplay.self_play`.  The ``TexasHoldem`` engine does not
expose a ``start`` method; the canonical entry point is
``TexasHoldem.play_game``.
"""

from __future__ import annotations

from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer
from poker_ai.engine.texas_holdem import TexasHoldem


def simulate_self_play(trainer: AICFRTrainer, num_games: int) -> None:
    """Run ``num_games`` of self-play against ``trainer``.

    .. deprecated::
        Prefer :class:`poker_ai.selfplay.self_play.SelfPlay` for new code.
    """
    for _ in range(num_games):
        # Disable verbose engine logging; no training entry points rely on it.
        game = TexasHoldem(num_players=2, verbose=False)
        # NOTE: ``TexasHoldem`` has no ``start`` method; this stub is
        # preserved only for import-compat.  Modern pipeline uses
        # ``play_game`` via ``SelfPlay``.
        game_state_sequence = game.start()  # type: ignore[attr-defined]
        # Simulate the game using the AI model and update strategies
        trainer.train(game_state_sequence)


if __name__ == "__main__":  # pragma: no cover - legacy entry point
    trainer = AICFRTrainer()
    simulate_self_play(trainer, num_games=10000)
    trainer.save_model()
