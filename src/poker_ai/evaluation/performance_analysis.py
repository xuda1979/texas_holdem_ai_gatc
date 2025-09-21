import glob
import os

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
        hole, community, history = prepare_transformer_input(
            game,
            player_index,
            self.max_seq_len,
            self.history_feature_dim,
            normalization_scale=normalization_scale,
        )
        advantages = (
            self.model(
                hole.unsqueeze(0).to(self.device),
                community.unsqueeze(0).to(self.device),
                history.unsqueeze(0).to(self.device),
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
        tournament_threshold: int = 10,
        tournament_size: int = 10,
        games_per_match: int = 10,
        device: str = "cpu",
    ):
        self.models_dir = models_dir
        if save_every_iterations is not None and save_every_iterations <= 0:
            save_every_iterations = None
        self.save_every_iterations = save_every_iterations
        self.tournament_threshold = tournament_threshold
        self.tournament_size = tournament_size
        self.games_per_match = games_per_match
        self.device = device

    def on_iteration_end(self, trainer, iteration: int) -> None:
        if self.save_every_iterations is None:
            return
        if iteration % self.save_every_iterations != 0:
            return
        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, f"model_{iteration}.pth")
        trainer.save_model(path)
        print(f"Model saved to {path} at iteration {iteration}")
        self._maybe_run_tournament()

    def _maybe_run_tournament(self) -> None:
        model_paths = sorted(glob.glob(os.path.join(self.models_dir, "*.pth")))
        if len(model_paths) < self.tournament_threshold:
            return
        selected = model_paths[-self.tournament_size :]
        print("Running model tournament...")
        results = run_tournament(selected, self.games_per_match, self.device)
        for model, score in results.items():
            print(f"{model}: {score}")
