import json
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from poker_ai.ai.model_loader import load_model_strategy
from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.gui.playStrategy import ModelAIStrategy, PlayerStrategy, RandomAIStrategy


@pytest.fixture
def cpu_only() -> None:
    """Mock ``torch.cuda.is_available`` to always return ``False``."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.mark.usefixtures("cpu_only")
@pytest.mark.parametrize("model_exists", [True, False])
def test_load_model_strategy(tmp_path: Path, model_exists: bool) -> None:
    model_path = tmp_path / "cfr_model.pth"
    if model_exists:
        config = {
            "input_feature_dim": 18,
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "num_actions": 4,
        }
        model = AdvantageNetwork(
            history_feature_dim=config["input_feature_dim"],
            card_feature_dim=config["input_feature_dim"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            num_actions=config["num_actions"],
        )
        torch.save(model.state_dict(), model_path)
        config_path = model_path.with_suffix(".config.json")
        config_path.write_text(json.dumps(config))
        strategy_cls: type[PlayerStrategy] = ModelAIStrategy
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy, device = load_model_strategy(str(model_path))
    else:
        strategy_cls = RandomAIStrategy
        with pytest.warns(RuntimeWarning):
            strategy, device = load_model_strategy(str(model_path))

    assert isinstance(strategy, strategy_cls)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


@pytest.mark.usefixtures("cpu_only")
@pytest.mark.parametrize(
    "config_writer, model_writer",
    [
        (None, torch.save),
        (lambda path, data: path.write_text(json.dumps(data)), None),
        (lambda path, data: path.write_text("invalid json"), torch.save),
    ],
)
def test_load_model_strategy_fallback(
    tmp_path: Path, config_writer, model_writer
) -> None:
    model_path = tmp_path / "cfr_model.pth"
    config_path = model_path.with_suffix(".config.json")
    config = {
        "input_feature_dim": 18,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_actions": 4,
    }
    model = AdvantageNetwork(
        history_feature_dim=config["input_feature_dim"],
        card_feature_dim=config["input_feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_actions=config["num_actions"],
    )

    if config_writer is not None:
        config_writer(config_path, config)

    if model_writer is not None:
        model_writer(model.state_dict(), model_path)

    with pytest.warns(RuntimeWarning):
        strategy, device = load_model_strategy(str(model_path))

    assert isinstance(strategy, RandomAIStrategy)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"
