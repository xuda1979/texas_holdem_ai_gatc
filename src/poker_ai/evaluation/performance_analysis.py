from __future__ import annotations

import glob
import logging
import os
from datetime import datetime

from poker_ai.model_storage import prepare_model_write_path, resolve_model_write_path

import torch

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import PlayerStrategy
from poker_ai.rules.cfr import calculate_strategy
from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import (
    infer_normalization_scale,
    prepare_transformer_input,
)


class EvalStrategy(PlayerStrategy):
    """Strategy wrapper used for automated evaluation."""

    def __init__(self, model_path: str, device: str):
        self.device = device

        payload = torch.load(model_path, map_location=self.device)
        metadata: dict[str, object] = {}
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            metadata = payload.get("metadata", {})  # type: ignore[assignment]
        else:
            state_dict = payload

        self.config: dict[str, object] = dict(metadata)
        self.history_feature_dim = int(metadata.get("history_feature_dim", 18))  # type: ignore[arg-type]
        self.card_feature_dim = int(metadata.get("card_feature_dim", self.history_feature_dim))  # type: ignore[arg-type]
        hidden_dim = int(
            metadata.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM)
        )  # type: ignore[arg-type]
        num_heads_raw = metadata.get("num_heads")
        if num_heads_raw is None:
            num_heads = AdvantageNetwork.recommended_num_heads(hidden_dim)
        else:
            num_heads = int(num_heads_raw)
        num_layers = int(
            metadata.get("num_layers", AdvantageNetwork.DEFAULT_NUM_LAYERS)
        )  # type: ignore[arg-type]
        self.num_actions = int(metadata.get("num_actions", 10))  # type: ignore[arg-type]
        self.max_seq_len = int(metadata.get("max_seq_len", 256))  # type: ignore[arg-type]

        self.model = AdvantageNetwork(
            history_feature_dim=self.history_feature_dim,
            card_feature_dim=self.card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=self.num_actions,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @property
    def is_human(self) -> bool:  # pragma: no cover - trivial
        return False

    @torch.no_grad()
    def choose_action(self, game: TexasHoldem, player_index: int):
        preferred_scale = None
        for key in ("normalization_scale", "chip_normalization", "starting_stack"):
            value = self.config.get(key)
            if isinstance(value, (int, float)):
                preferred_scale = float(value)
                break

        normalization_scale = infer_normalization_scale(game, preferred_scale)
        hole, community, history, mask = prepare_transformer_input(
            game,
            player_index,
            self.max_seq_len,
            self.history_feature_dim,
            normalization_scale=normalization_scale,
            return_mask=True,
        )
        mask = mask.to(torch.bool)
        hole_batch = hole.unsqueeze(0).to(self.device)
        community_batch = community.unsqueeze(0).to(self.device)
        history_batch = history.unsqueeze(0).to(self.device)
        key_padding_mask = (~mask.unsqueeze(0)).to(self.device)
        advantages = (
            self.model(
                hole_batch,
                community_batch,
                history_batch,
                key_padding_mask=key_padding_mask,
            )
            .squeeze(0)
            .cpu()
        )
        legal_mask = get_legal_actions_mask(game, player_index, self.num_actions)
        advantages[~legal_mask] = -1e9
        policy = calculate_strategy(advantages, self.num_actions)
        if policy.sum() > 0:
            action_idx = torch.multinomial(policy, 1).item()
        else:  # pragma: no cover - fallback
            valid_indices = torch.where(legal_mask)[0]
            action_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
        action = get_action_from_index(action_idx, game, player_id=player_index)
        return action_to_tuple(action)


def run_tournament(
    model_paths: list[str], games_per_match: int = 10, device: str = "cpu"
) -> dict[str, int]:
    """Run a simple round-robin tournament between models.

    Returns a mapping of model path to number of games won."""

    scores: dict[str, int] = {p: 0 for p in model_paths}
    if len(model_paths) < 2:
        return scores

    for i, path_i in enumerate(model_paths):
        for j, path_j in enumerate(model_paths[i + 1 :], start=i + 1):
            strat_i = EvalStrategy(path_i, device)
            strat_j = EvalStrategy(path_j, device)
            game = TexasHoldem(
                num_players=2,
                starting_stack=1000,
                player_strategies=[strat_i, strat_j],
                verbose=False,
            )
            player_paths = [path_i, path_j]
            for _ in range(games_per_match):
                previous_chips = list(game.rules.player_chips)
                game.play_game()
                updated_chips = list(game.rules.player_chips)
                deltas = [after - before for before, after in zip(previous_chips, updated_chips)]

                positive_winners = [idx for idx, delta in enumerate(deltas) if delta > 0]
                winner_index: int | None = None

                if len(positive_winners) == 1:
                    winner_index = positive_winners[0]
                elif len(positive_winners) == 0:
                    winner_info = getattr(game, "last_winner", None)
                    if isinstance(winner_info, int):
                        winner_index = winner_info
                    elif isinstance(winner_info, list) and len(winner_info) == 1:
                        winner_index = winner_info[0]

                if (
                    winner_index is not None
                    and 0 <= winner_index < len(player_paths)
                    and deltas[winner_index] > 0
                ):
                    scores[player_paths[winner_index]] += 1
    return scores


class ModelPerformanceAnalyzer:
    """Utility to periodically save models and evaluate them via tournaments."""

    def __init__(
        self,
        models_dir: str = "models",
        save_every_iterations: int | None = 100000,
        tournament_threshold: int = 20,
        tournament_size: int | None = None,
        games_per_match: int = 10,
        device: str = "cpu",
        *,
        max_no_improvement_samples: int = 200_000,
    ):
        self.models_dir = models_dir
        if save_every_iterations is not None and save_every_iterations <= 0:
            save_every_iterations = None
        self.save_every_iterations = save_every_iterations
        if tournament_size is None:
            tournament_size = tournament_threshold
        self.max_models = max(1, int(tournament_size))
        self.games_per_match = games_per_match
        self.device = device
        self.max_no_improvement_samples = max_no_improvement_samples
        self._no_improvement_samples = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def no_improvement_samples(self) -> int:
        return self._no_improvement_samples

    def on_iteration_end(self, trainer, iteration: int) -> bool:
        if self.save_every_iterations is None:
            return True
        if iteration % self.save_every_iterations != 0:
            return True
        resolve_model_write_path(self.models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        path = self._save_snapshot(trainer)
        self.logger.info("Model saved to %s at iteration %s", path, iteration)
        keep_new = self._evaluate_new_model(path)
        if keep_new:
            self._no_improvement_samples = 0
        else:
            if self.save_every_iterations is not None:
                self._no_improvement_samples += self.save_every_iterations
            if (
                self.max_no_improvement_samples > 0
                and self._no_improvement_samples >= self.max_no_improvement_samples
            ):
                self.logger.warning(
                    "Stopping training | no improvement after %s samples",
                    self._no_improvement_samples,
                )
                return False
        return True

    def _save_snapshot(self, trainer) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"model_{timestamp}.pth"
        path = str(prepare_model_write_path(os.path.join(self.models_dir, filename)))
        trainer.save_model(path)
        return path

    def _evaluate_new_model(self, new_model_path: str) -> bool:
        model_paths = sorted(glob.glob(os.path.join(self.models_dir, "*.pth")))
        if len(model_paths) <= self.max_models:
            self.logger.info(
                "Model pool size %s/%s; retaining new snapshot %s",
                len(model_paths),
                self.max_models,
                new_model_path,
            )
            return True

        self.logger.info(
            "Running model tournament with %s candidates", len(model_paths)
        )
        results = run_tournament(model_paths, self.games_per_match, self.device)
        for model, score in results.items():
            self.logger.info("Tournament result | model=%s | wins=%s", model, score)

        scored_models = [
            (
                results.get(path, 0),
                os.path.getmtime(path),
                path,
            )
            for path in model_paths
        ]
        scored_models.sort(key=lambda item: (-item[0], -item[1], item[2]))
        survivors = {path for _, _, path in scored_models[: self.max_models]}
        for path in model_paths:
            if path not in survivors:
                try:
                    os.remove(path)
                    self.logger.info("Removed model %s from pool", path)
                except OSError:
                    self.logger.exception("Failed to remove model %s", path)

        if new_model_path in survivors:
            self.logger.info(
                "New model %s retained in top %s", new_model_path, self.max_models
            )
            return True

        self.logger.info(
            "New model %s discarded after tournament", new_model_path
        )
        return False
