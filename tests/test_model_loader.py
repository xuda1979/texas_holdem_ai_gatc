import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from poker_ai.ai.model_loader import load_model_strategy
from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer
from poker_ai.gui.playStrategy import ModelAIStrategy, RandomAIStrategy


@pytest.fixture
def cpu_only() -> None:
    """Mock ``torch.cuda.is_available`` to always return ``False``."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.mark.usefixtures("cpu_only")
@pytest.mark.parametrize("model_exists", [True, False])
def test_load_model_strategy(tmp_path: Path, model_exists: bool) -> None:
    model_path = tmp_path / "cfr_model.pth"
    metadata = {
        "history_feature_dim": 18,
        "card_feature_dim": 17,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_actions": 4,
        "max_seq_len": 32,
    }
    if model_exists:
        model = AdvantageNetwork(
            history_feature_dim=metadata["history_feature_dim"],
            card_feature_dim=metadata["card_feature_dim"],
            hidden_dim=metadata["hidden_dim"],
            num_heads=metadata["num_heads"],
            num_layers=metadata["num_layers"],
            num_actions=metadata["num_actions"],
        )
        payload = {"state_dict": model.state_dict(), "metadata": metadata}
        torch.save(payload, model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            strategy, device = load_model_strategy(str(model_path))
        assert isinstance(strategy, ModelAIStrategy)
        assert strategy.config["history_feature_dim"] == metadata["history_feature_dim"]
        assert strategy.config["card_feature_dim"] == metadata["card_feature_dim"]
    else:
        with pytest.warns(RuntimeWarning):
            strategy, device = load_model_strategy(str(model_path))
        assert isinstance(strategy, RandomAIStrategy)

    assert isinstance(device, torch.device)
    assert device.type == "cpu"


@pytest.mark.usefixtures("cpu_only")
@pytest.mark.parametrize(
    "payload_factory",
    [
        lambda state: state,
        lambda state: {},
        lambda state: {"state_dict": state},
        lambda state: {"metadata": {"history_feature_dim": 18, "num_actions": 4}},
        lambda state: {
            "state_dict": state,
            "metadata": {"history_feature_dim": 18, "num_actions": 4},
        },
        lambda state: {"state_dict": state, "metadata": "invalid"},
    ],
)
def test_load_model_strategy_fallback(
    tmp_path: Path, payload_factory
) -> None:
    model_path = tmp_path / "cfr_model.pth"
    metadata = {
        "history_feature_dim": 18,
        "card_feature_dim": 17,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_actions": 4,
    }
    model = AdvantageNetwork(
        history_feature_dim=metadata["history_feature_dim"],
        card_feature_dim=metadata["card_feature_dim"],
        hidden_dim=metadata["hidden_dim"],
        num_heads=metadata["num_heads"],
        num_layers=metadata["num_layers"],
        num_actions=metadata["num_actions"],
    )

    payload = payload_factory(model.state_dict())
    torch.save(payload, model_path)

    with pytest.warns(RuntimeWarning):
        strategy, device = load_model_strategy(str(model_path))

    assert isinstance(strategy, RandomAIStrategy)
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


@pytest.mark.usefixtures("cpu_only")
def test_load_model_strategy_trainer_payload(tmp_path: Path) -> None:
    model_path = tmp_path / "trainer_model.pth"
    trainer = SingleNetworkCFRTrainer(input_feature_dim=18, hidden_dim=32, num_actions=4, lr=1e-3)
    trainer.save_model(str(model_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        strategy, device = load_model_strategy(str(model_path))

    assert isinstance(strategy, ModelAIStrategy)
    assert strategy.config["history_feature_dim"] == trainer.history_feature_dim
    assert strategy.config["card_feature_dim"] == trainer.card_feature_dim
    assert isinstance(device, torch.device)
    assert device.type == "cpu"
