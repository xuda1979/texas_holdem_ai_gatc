from __future__ import annotations

# ruff: noqa: ANN001,ANN101,ANN201,ANN204
import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

import torch
import torch.nn.functional as F

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.config import config, load_config
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import (
    infer_normalization_scale,
    prepare_transformer_input,
)
from poker_ai.utils.model_paths import find_latest_model_checkpoint


class TransformerStrategy:
    """Wraps an ``AdvantageNetwork`` with game logic helpers."""

    def __init__(self, model: AdvantageNetwork, config: dict, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self._scale_hint = None
        for key in ("normalization_scale", "chip_normalization", "starting_stack"):
            value = self.config.get(key)
            if isinstance(value, (int, float)):
                self._scale_hint = float(value)
                break

    @property
    def is_human(self) -> bool:
        return False

    @torch.no_grad()
    def choose_action(self, game: TexasHoldem, player_index: int):
        max_seq_len = self.config.get("max_seq_len", 256)
        d_raw_feature = self.config.get("d_raw_feature", self.config.get("input_feature_dim", 18))
        normalization_scale = infer_normalization_scale(game, self._scale_hint)
        hole, community, history, mask = prepare_transformer_input(
            cast(Any, game),
            player_index,
            max_seq_len,
            d_raw_feature,
            normalization_scale=normalization_scale,
            return_mask=True,
        )
        mask = mask.to(torch.bool)
        hole_batch = hole.unsqueeze(0).to(self.device)
        community_batch = community.unsqueeze(0).to(self.device)
        history_batch = history.unsqueeze(0).to(self.device)
        padding_mask = (~mask).unsqueeze(0).to(self.device)
        advantages = (
            self.model(
                hole_batch,
                community_batch,
                history_batch,
                padding_mask=padding_mask,
            )
            .squeeze(0)
            .cpu()
        )
        num_actions = self.config.get("num_actions", self.model.num_actions)
        legal_mask = get_legal_actions_mask(game, player_index, num_actions)
        temperature = float(self.config.get("self_play", {}).get("temperature", 1.0))
        epsilon = float(self.config.get("self_play", {}).get("epsilon", 0.05))

        masked_advantages = advantages.clone()
        masked_advantages[~legal_mask] = -float("inf")

        tau = max(temperature, 1e-6)
        logits = masked_advantages / tau
        policy = F.softmax(logits, dim=-1)

        if legal_mask.any():
            uniform = legal_mask.float() / legal_mask.float().sum()
            policy = (1.0 - epsilon) * policy + epsilon * uniform
            policy = policy * legal_mask.float()
            total = policy.sum()
            if total.item() == 0:
                policy = uniform
            else:
                policy = policy / total
        else:  # pragma: no cover - defensive fallback
            policy = torch.full_like(advantages, 1.0 / len(advantages))

        action_idx = torch.multinomial(policy, 1).item()
        action = get_action_from_index(action_idx, game, player_index)
        return action_to_tuple(action)


def parse_args():
    parser = argparse.ArgumentParser(description="Run self-play simulation")
    parser.add_argument("--config", default=None, help="Path to configuration YAML file")
    return parser.parse_args()


def load_transformer_model(cfg):
    model_name = cfg.get("model", {}).get("name", "texas_holdem_transformer_ai")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_path = get_model_path(model_name)

    candidate_path: str | None = None
    last_mtime: float | None = None

    if os.path.exists(default_path):
        candidate_path = default_path
        last_mtime = os.path.getmtime(default_path)
    else:
        latest = find_latest_model_checkpoint(prefix=model_name)
        if latest is None:
            latest = find_latest_model_checkpoint()
        if latest is not None:
            candidate_path, last_mtime = latest

    if candidate_path is not None:
        try:
            strategy = load_existing_model(candidate_path, device)
            return strategy, candidate_path, last_mtime
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to load existing model: {exc}. Initializing a new model.")

    strategy = initialize_new_model(device)
    return strategy, default_path, None


def get_model_path(model_name):
    return os.path.join(config.MODEL_DIR, f"{model_name}.pth")


def load_existing_model(weights_path, device):
    metadata, state_dict = _load_state_and_metadata(weights_path, device)
    network_params = {
        "history_feature_dim": metadata["history_feature_dim"],
        "card_feature_dim": metadata["card_feature_dim"],
        "hidden_dim": metadata["hidden_dim"],
        "num_heads": metadata["num_heads"],
        "num_layers": metadata["num_layers"],
        "num_actions": metadata["num_actions"],
    }
    model = AdvantageNetwork(**network_params)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {weights_path}")
    return TransformerStrategy(model, metadata, device)


def initialize_new_model(device):
    print("No existing model found. Initializing a new model.")
    model_config = {
        "history_feature_dim": 18,
        "card_feature_dim": 17,
        "hidden_dim": AdvantageNetwork.DEFAULT_HIDDEN_DIM,
        "num_heads": AdvantageNetwork.DEFAULT_NUM_HEADS,
        "num_layers": AdvantageNetwork.DEFAULT_NUM_LAYERS,
        "num_actions": 10,
        "max_seq_len": 256,
        "d_raw_feature": 18,
        "input_feature_dim": 18,
        "d_card_feature": 17,
    }
    network_params = {
        "history_feature_dim": model_config["history_feature_dim"],
        "card_feature_dim": model_config["card_feature_dim"],
        "hidden_dim": model_config["hidden_dim"],
        "num_heads": model_config["num_heads"],
        "num_layers": model_config["num_layers"],
        "num_actions": model_config["num_actions"],
    }
    model = AdvantageNetwork(**network_params)
    return TransformerStrategy(model, model_config, device)


def save_transformer_model(transformer_strategy):
    models_dir = config.MODEL_DIR
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    weight_path = get_save_path()

    save_weights(transformer_strategy, weight_path)


def get_save_path():
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    return os.path.join(config.MODEL_DIR, f"{timestamp}_model.pth")


def save_weights(transformer_strategy, weight_path):
    try:
        metadata = _metadata_from_config(transformer_strategy.config)
        payload = {
            "state_dict": transformer_strategy.model.state_dict(),
            "metadata": metadata,
        }
        torch.save(payload, weight_path)
        print(f"Saved model weights to {weight_path}")
    except Exception as e:
        print(f"Error saving model weights: {e}")


def reload_weights_if_updated(transformer_strategy, weights_path, last_mtime):
    """Reload to the most recent checkpoint if a newer file is available."""
    latest = find_latest_model_checkpoint()
    if latest is None:
        return weights_path, last_mtime

    latest_path, latest_mtime = latest
    should_reload = (
        weights_path != latest_path
        or last_mtime is None
        or latest_mtime > last_mtime
    )

    if not should_reload:
        return weights_path, last_mtime

    try:
        metadata, state = _load_state_and_metadata(latest_path, transformer_strategy.device)
        transformer_strategy.model.load_state_dict(state)
        transformer_strategy.config.update(metadata)
        print(
            f"[Model Reloaded] {datetime.now().isoformat()} - "
            f"Loaded weights from {latest_path}"
        )
        return latest_path, latest_mtime
    except Exception as e:
        print(f"[Model Reloaded] Failed to reload weights: {e}")
        return weights_path, last_mtime


def _metadata_from_config(config_dict: Mapping[str, Any]) -> dict[str, Any]:
    history = int(
        config_dict.get(
            "history_feature_dim",
            config_dict.get("d_raw_feature", config_dict.get("input_feature_dim", 18)),
        )
    )
    card = int(config_dict.get("card_feature_dim", config_dict.get("d_card_feature", 17)))
    hidden = int(config_dict.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM))
    num_heads_value = config_dict.get("num_heads")
    if num_heads_value is None:
        num_heads = AdvantageNetwork.recommended_num_heads(hidden)
    else:
        num_heads = int(num_heads_value)
    num_layers = int(
        config_dict.get("num_layers", AdvantageNetwork.DEFAULT_NUM_LAYERS)
    )
    num_actions = int(config_dict.get("num_actions", 10))
    max_seq_len = int(config_dict.get("max_seq_len", 256))

    metadata: dict[str, Any] = {
        "history_feature_dim": history,
        "card_feature_dim": card,
        "hidden_dim": hidden,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "num_actions": num_actions,
        "max_seq_len": max_seq_len,
    }

    trainer = config_dict.get("trainer")
    if trainer is not None:
        metadata["trainer"] = trainer

    return metadata


def _load_state_and_metadata(
    path: str, device: torch.device
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("Model file is missing metadata payload")

    metadata_obj = payload.get("metadata")
    if not isinstance(metadata_obj, Mapping):
        raise ValueError("Model metadata missing or malformed")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Model payload missing state_dict")

    metadata = _normalize_metadata(metadata_obj)
    return metadata, state_dict


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    try:
        history_feature_dim = int(metadata["history_feature_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid model metadata: missing history_feature_dim") from exc

    try:
        num_actions = int(metadata["num_actions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid model metadata: missing num_actions") from exc

    card_feature_dim = int(metadata.get("card_feature_dim", history_feature_dim))
    hidden_dim = int(
        metadata.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM)
    )
    num_heads_value = metadata.get("num_heads")
    if num_heads_value is None:
        num_heads = AdvantageNetwork.recommended_num_heads(hidden_dim)
    else:
        num_heads = int(num_heads_value)
    num_layers = int(
        metadata.get("num_layers", AdvantageNetwork.DEFAULT_NUM_LAYERS)
    )
    max_seq_len = int(metadata.get("max_seq_len", 256))

    normalized = dict(metadata)
    normalized["history_feature_dim"] = history_feature_dim
    normalized["card_feature_dim"] = card_feature_dim
    normalized["hidden_dim"] = hidden_dim
    normalized["num_heads"] = num_heads
    normalized["num_layers"] = num_layers
    normalized["num_actions"] = num_actions
    normalized["max_seq_len"] = max_seq_len
    normalized.setdefault("d_raw_feature", history_feature_dim)
    normalized.setdefault("input_feature_dim", history_feature_dim)
    normalized.setdefault("d_card_feature", card_feature_dim)

    return normalized
def save_game_history(game):
    history_dir = os.path.join(config.BASE_DATA_DIR, "historical_actions")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    filepath = get_history_filepath()
    save_game_to_file(game, filepath)


def get_history_filepath():
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    return os.path.join(
        config.BASE_DATA_DIR, "historical_actions", f"{timestamp}_simulated_game.json"
    )


def save_game_to_file(game, filepath):
    game_data = extract_game_data(game)
    try:
        with open(filepath, "w") as f:
            json.dump(game_data, f, indent=4)
        print(f"Saved simulated game history to {filepath}")
    except Exception as e:
        print(f"Error saving game history: {e}")


def extract_game_data(game):
    return {
        "hand_number": game.hand_count,
        "dealer": game.rules.dealer_button + 1,
        "actions": game.rules.betting_history,
        "community_cards": game.rules.community_cards,
        "pot": game.rules.pot,
        "players": game.get_player_status(),
    }


def simulate_game(transformer_strategy):
    num_players = random.randint(2, 10)
    starting_stack = 10000
    print(f"\n--- Starting game with {num_players} AI players ---")

    player_strategies = [transformer_strategy] * num_players
    game = TexasHoldem(num_players, starting_stack, player_strategies)
    game.play_game()
    save_game_history(game)


def periodic_save(transformer_strategy, interval=1800):
    save_thread = threading.Thread(
        target=save_loop, args=(transformer_strategy, interval), daemon=True
    )
    save_thread.start()
    print(f"Started periodic model saving every {interval / 60} minutes.")


def save_loop(transformer_strategy, interval):
    while True:
        time.sleep(interval)
        print("\n[Periodic Save] Saving Transformer model...")
        save_transformer_model(transformer_strategy)
        print("[Periodic Save] Model saved successfully.\n")


def handle_termination(transformer_strategy):
    signal.signal(signal.SIGINT, lambda sig, frame: terminate_gracefully(transformer_strategy))
    signal.signal(signal.SIGTERM, lambda sig, frame: terminate_gracefully(transformer_strategy))
    print("Signal handlers for termination set up.")


def terminate_gracefully(transformer_strategy):
    print("\n[Termination] Saving model before exit...")
    save_transformer_model(transformer_strategy)
    print("[Termination] Model saved. Exiting now.")
    sys.exit(0)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    transformer_strategy, weights_path, last_mtime = load_transformer_model(cfg)
    interval = cfg.get("self_play", {}).get("save_interval", 1800)
    periodic_save(transformer_strategy, interval=interval)
    handle_termination(transformer_strategy)

    print("Starting self-play simulation. Press Ctrl+C to terminate.")
    while True:
        weights_path, last_mtime = reload_weights_if_updated(
            transformer_strategy, weights_path, last_mtime
        )
        simulate_game(transformer_strategy)
        time.sleep(1)


if __name__ == "__main__":
    main()
