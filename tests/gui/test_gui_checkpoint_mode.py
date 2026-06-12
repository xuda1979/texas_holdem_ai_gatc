from __future__ import annotations

from unittest.mock import MagicMock, patch

from poker_ai.ai.model_loader import DEFAULT_MODEL_PATH
from poker_ai.gui.gui import GUIHumanStrategy, PokerGameGUI


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


def test_start_new_game_uses_checkpoint_model_for_ai_players() -> None:
    gui = _build_gui()
    gui.total_players_var = _Var("3")
    gui.starting_stack_var = _Var("5000")
    gui.ai_opponent_mode_var = _Var("checkpoint")
    gui.model_path_var = _Var("/tmp/checkpoints/final_model.pth")

    fake_strategy = MagicMock(name="checkpoint_strategy")

    with (
        patch("poker_ai.ai.model_loader.load_model_strategy", return_value=(fake_strategy, "cpu")) as mock_loader,
        patch("poker_ai.gui.gui.TexasHoldem") as mock_game,
        patch.object(gui, "setup_game_gui") as mock_setup_game_gui,
        patch.object(gui, "play_hand") as mock_play_hand,
    ):
        gui.start_new_game()

    mock_loader.assert_called_once_with("/tmp/checkpoints/final_model.pth")
    mock_game.assert_called_once()
    args, _kwargs = mock_game.call_args
    assert args[0] == 3
    assert args[1] == 5000
    assert isinstance(args[2][0], GUIHumanStrategy)
    assert args[2][1:] == [fake_strategy, fake_strategy]
    mock_setup_game_gui.assert_called_once()
    mock_play_hand.assert_called_once()


def test_start_new_game_uses_default_checkpoint_when_model_path_blank() -> None:
    gui = _build_gui()
    gui.total_players_var = _Var("2")
    gui.starting_stack_var = _Var("1000")
    gui.ai_opponent_mode_var = _Var("checkpoint")
    gui.model_path_var = _Var("   ")

    fake_strategy = MagicMock(name="checkpoint_strategy")

    with (
        patch("poker_ai.ai.model_loader.load_model_strategy", return_value=(fake_strategy, "cpu")) as mock_loader,
        patch("poker_ai.gui.gui.TexasHoldem"),
        patch.object(gui, "setup_game_gui"),
        patch.object(gui, "play_hand"),
    ):
        gui.start_new_game()

    mock_loader.assert_called_once_with(DEFAULT_MODEL_PATH)
