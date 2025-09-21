import warnings

import torch
from collections.abc import Mapping

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.gui.playStrategy import ModelAIStrategy, PlayerStrategy, RandomAIStrategy

DEFAULT_MODEL_PATH = "trained_models/cfr_model.pth"


def load_model_strategy(
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple[PlayerStrategy, torch.device]:
    """Load an AI strategy from ``model_path``.

    If the model file is missing or fails to load, a :class:`RandomAIStrategy`
    is returned instead and a ``RuntimeWarning`` is emitted. The model is
    loaded onto ``CUDA`` when available, otherwise ``CPU``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        payload = torch.load(model_path, map_location=device)

        if not isinstance(payload, Mapping):
            raise ValueError("Model payload missing metadata")

        metadata_obj = payload.get("metadata")
        if not isinstance(metadata_obj, Mapping):
            raise ValueError("Model metadata missing or malformed")

        try:
            history_feature_dim = int(metadata_obj["history_feature_dim"])
            card_feature_dim = int(metadata_obj["card_feature_dim"])
            num_actions = int(metadata_obj["num_actions"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model metadata: {exc}") from exc

        hidden_dim = int(
            metadata_obj.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM)
        )
        num_heads_value = metadata_obj.get("num_heads")
        if num_heads_value is None:
            num_heads = AdvantageNetwork.recommended_num_heads(hidden_dim)
        else:
            num_heads = int(num_heads_value)
        num_layers = int(
            metadata_obj.get("num_layers", AdvantageNetwork.DEFAULT_NUM_LAYERS)
        )
        max_seq_len = int(metadata_obj.get("max_seq_len", 256))

        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Model payload missing state_dict")

        model = AdvantageNetwork(
            history_feature_dim=history_feature_dim,
            card_feature_dim=card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=num_actions,
        )

        model.load_state_dict(state_dict)

        config = dict(metadata_obj)
        config["history_feature_dim"] = history_feature_dim
        config["card_feature_dim"] = card_feature_dim
        config["hidden_dim"] = hidden_dim
        config["num_heads"] = num_heads
        config["num_layers"] = num_layers
        config["num_actions"] = num_actions
        config["max_seq_len"] = max_seq_len
        config.setdefault("input_feature_dim", history_feature_dim)
        config.setdefault("d_raw_feature", history_feature_dim)

        return ModelAIStrategy(model, config, device), device
    except (FileNotFoundError, ValueError) as exc:
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
