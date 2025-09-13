import json
import os
import warnings

import torch

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.gui.playStrategy import ModelAIStrategy, PlayerStrategy, RandomAIStrategy

DEFAULT_MODEL_PATH = "trained_models/cfr_model.pth"


def load_model_strategy(
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple[PlayerStrategy, torch.device]:
    """Load an AI strategy from ``model_path``.

    If the model file or its companion ``.config.json`` file is missing or
    fails to load, a :class:`RandomAIStrategy` is returned instead and a
    ``RuntimeWarning`` is emitted. The model is loaded onto ``CUDA`` when
    available, otherwise ``CPU``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.splitext(model_path)[0] + ".config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)

        model = AdvantageNetwork(
            history_feature_dim=config["input_feature_dim"],
            card_feature_dim=config["input_feature_dim"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            num_actions=config["num_actions"],
        )
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return ModelAIStrategy(model, config, device), device
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Failed to load model from {model_path}: {exc}. Falling back to RandomAIStrategy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return RandomAIStrategy(), device
    except Exception as exc:  # pragma: no cover - unexpected error
        warnings.warn(
            f"Unexpected error loading model from {model_path}: {exc}. "
            "Falling back to RandomAIStrategy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return RandomAIStrategy(), device
