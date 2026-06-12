from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.gui.gui import GUIHumanStrategy, PokerGameGUI
from poker_ai.gui.playStrategy import ModelAIStrategy, RandomAIStrategy


class _Var:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


def _build_gui() -> PokerGameGUI:
    with (
        patch("poker_ai.gui.gui.tk") as mock_tk,
        patch("poker_ai.gui.gui.messagebox"),
        patch.object(PokerGameGUI, "_load_card_images"),
        patch.object(PokerGameGUI, "setup_initial_gui"),
    ):
        mock_tk.Tk.return_value.mainloop = MagicMock()
        return PokerGameGUI()


def _write_checkpoint(model_path: Path) -> None:
    metadata = {
        "history_feature_dim": 18,
        "card_feature_dim": 17,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_actions": 4,
        "max_seq_len": 32,
    }
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


@patch("torch.cuda.is_available", return_value=False)
def test_start_new_game_loads_real_checkpoint_strategy(_mock_cuda, tmp_path: Path) -> None:
    gui = _build_gui()
    checkpoint_path = tmp_path / "beautiful_gui_checkpoint.pth"
    _write_checkpoint(checkpoint_path)

    gui.total_players_var = _Var("3")
    gui.starting_stack_var = _Var("5000")
    gui.ai_opponent_mode_var = _Var("checkpoint")
    gui.model_path_var = _Var(str(checkpoint_path))

    with (
        patch("poker_ai.gui.gui.TexasHoldem") as mock_game,
        patch.object(gui, "setup_game_gui") as mock_setup_game_gui,
        patch.object(gui, "play_hand") as mock_play_hand,
    ):
        gui.start_new_game()

    args, _kwargs = mock_game.call_args
    strategies = args[2]
    assert isinstance(strategies[0], GUIHumanStrategy)
    assert len(strategies) == 3
    assert all(isinstance(strategy, ModelAIStrategy) for strategy in strategies[1:])
    assert gui.current_model_path == str(checkpoint_path)
    assert checkpoint_path.name in gui.ai_summary_text
    assert "Checkpoint model" in gui.ai_summary_text
    mock_setup_game_gui.assert_called_once()
    mock_play_hand.assert_called_once()


@patch("torch.cuda.is_available", return_value=False)
def test_start_new_game_reports_checkpoint_fallback_when_missing(_mock_cuda, tmp_path: Path) -> None:
    gui = _build_gui()
    missing_checkpoint = tmp_path / "missing_checkpoint.pth"

    gui.total_players_var = _Var("2")
    gui.starting_stack_var = _Var("1000")
    gui.ai_opponent_mode_var = _Var("checkpoint")
    gui.model_path_var = _Var(str(missing_checkpoint))

    with (
        pytest.warns(RuntimeWarning, match="Falling back to RandomAIStrategy"),
        patch("poker_ai.gui.gui.TexasHoldem") as mock_game,
        patch.object(gui, "setup_game_gui"),
        patch.object(gui, "play_hand"),
    ):
        gui.start_new_game()

    args, _kwargs = mock_game.call_args
    strategies = args[2]
    assert isinstance(strategies[1], RandomAIStrategy)
    assert "fallback random" in gui.ai_summary_text.lower()
    assert missing_checkpoint.name in gui.ai_summary_text
