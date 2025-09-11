import glob
import os
from typing import Dict, List

import torch

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import PlayerStrategy
from poker_ai.rules.cfr import calculate_strategy
from poker_ai.utils.action_mapping import get_action_from_index, get_legal_actions_mask
from poker_ai.utils.state_representation import prepare_transformer_input


class EvalStrategy(PlayerStrategy):
    """Strategy wrapper used for automated evaluation."""

    def __init__(self, model_path: str, device: str, num_actions: int = 10):
        self.device = device
        self.num_actions = num_actions

        self.model = AdvantageNetwork(
            history_feature_dim=18,
            card_feature_dim=17,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            num_actions=self.num_actions,
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @property
    def is_human(self) -> bool:  # pragma: no cover - trivial
        return False

    @torch.no_grad()
    def choose_action(self, game: TexasHoldem, player_index: int):
        hole, community, history = prepare_transformer_input(game, player_index, 256, 18)
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
        return get_action_from_index(action_idx, game, player_id=player_index)


def run_tournament(model_paths: List[str], games_per_match: int = 10, device: str = "cpu") -> Dict[str, int]:
    """Run a simple round-robin tournament between models.

    Returns a mapping of model path to number of games won."""

    scores: Dict[str, int] = {p: 0 for p in model_paths}
    if len(model_paths) < 2:
        return scores

    for i, path_i in enumerate(model_paths):
        for j, path_j in enumerate(model_paths[i + 1 :], start=i + 1):
            strat_i = EvalStrategy(path_i, device)
            strat_j = EvalStrategy(path_j, device)
            for _ in range(games_per_match):
                game = TexasHoldem(
                    num_players=2,
                    starting_stack=1000,
                    player_strategies=[strat_i, strat_j],
                    verbose=False,
                )
                game.play_game()
                chips = game.rules.player_chips
                if chips[0] > chips[1]:
                    scores[path_i] += 1
                elif chips[1] > chips[0]:
                    scores[path_j] += 1
    return scores


class ModelPerformanceAnalyzer:
    """Utility to periodically save models and evaluate them via tournaments."""

    def __init__(
        self,
        models_dir: str = "models",
        save_every_samples: int = 100000,
        tournament_threshold: int = 10,
        tournament_size: int = 10,
        games_per_match: int = 10,
        device: str = "cpu",
    ):
        self.models_dir = models_dir
        self.save_every_samples = save_every_samples
        self.tournament_threshold = tournament_threshold
        self.tournament_size = tournament_size
        self.games_per_match = games_per_match
        self.device = device

    def on_iteration_end(self, trainer, sample_count: int) -> None:
        if sample_count % self.save_every_samples != 0:
            return
        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, f"model_{sample_count}.pth")
        trainer.save_model(path)
        print(f"Model saved to {path}")
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
