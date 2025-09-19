"""Thin wrapper exposing GUI classes with player-selection helpers."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import poker_ai.gui.gui as _gui


class PokerGameGUI(_gui.PokerGameGUI):
    """Subclass that layers a simple player-selection screen on top."""

    def __init__(self, *args, **kwargs):  # pragma: no cover - GUI wiring
        # allow tests to monkeypatch ``play.gui.tk`` which then flows into the
        # underlying module before initialization
        _gui.tk = tk
        self.selection_frame: tk.Frame | None = None
        self.ai_count_var: tk.StringVar | None = None
        self.starting_stack_var: tk.StringVar | None = None
        # Explicitly document the expected human index for validate script
        self.human_player_index = 0
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Player selection helpers
    # ------------------------------------------------------------------
    def setup_gui(self):  # pragma: no cover - exercised via integration tests
        """Display the player selection dialog instead of the base layout."""
        self.show_player_selection()

    def show_player_selection(self):  # pragma: no cover - GUI wiring
        """Show a modal-style frame that lets users pick AI opponents."""
        if self.selection_frame is not None:
            self.selection_frame.destroy()

        # Hide existing widgets but keep them around so they can be restored
        for widget in list(self.root.winfo_children()):
            widget.pack_forget()

        self.selection_frame = tk.Frame(self.root, bg="darkgreen")
        self.selection_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)

        title_label = tk.Label(
            self.selection_frame,
            text="Choose Opponents",
            font=("Arial", 24, "bold"),
            bg="darkgreen",
            fg="white",
        )
        title_label.pack(pady=(0, 20))

        instructions = tk.Label(
            self.selection_frame,
            text=(
                "Enter how many AI opponents you would like to face. "
                "You are always the only human player."
            ),
            font=("Arial", 12),
            bg="darkgreen",
            fg="white",
            wraplength=600,
            justify=tk.LEFT,
        )
        instructions.pack(pady=(0, 20))

        spinbox_frame = tk.Frame(self.selection_frame, bg="darkgreen")
        spinbox_frame.pack(pady=10)

        tk.Label(
            spinbox_frame,
            text="Number of AI opponents (1-8):",
            font=("Arial", 14),
            bg="darkgreen",
            fg="white",
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.ai_count_var = tk.StringVar(value="3")
        tk.Spinbox(
            spinbox_frame,
            from_=1,
            to=8,
            width=5,
            font=("Arial", 14),
            textvariable=self.ai_count_var,
            justify="center",
        ).pack(side=tk.LEFT)

        stack_frame = tk.Frame(self.selection_frame, bg="darkgreen")
        stack_frame.pack(pady=30)

        tk.Label(
            stack_frame,
            text="Starting stack:",
            font=("Arial", 14),
            bg="darkgreen",
            fg="white",
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.starting_stack_var = tk.StringVar(value="10000")
        stack_options = ["1000", "5000", "10000", "20000", "50000"]
        stack_menu = tk.OptionMenu(stack_frame, self.starting_stack_var, *stack_options)
        stack_menu.config(font=("Arial", 14))
        stack_menu.pack(side=tk.LEFT)

        start_button = tk.Button(
            self.selection_frame,
            text="Start Game",
            font=("Arial", 16, "bold"),
            bg="red",
            fg="white",
            command=self.start_game_from_selection,
            width=18,
            height=2,
        )
        start_button.pack(pady=40)

    def start_game_from_selection(self):  # pragma: no cover - GUI wiring
        """Validate numeric input and bootstrap the requested match."""

        if self.ai_count_var is None or self.starting_stack_var is None:
            messagebox.showerror("Invalid Input", "Please open the player selection screen first.")
            return

        try:
            ai_count = int(self.ai_count_var.get())
        except (TypeError, ValueError, tk.TclError):
            messagebox.showerror("Invalid Input", "Please enter a number between 1 and 8.")
            return

        if ai_count < 1 or ai_count > 8:
            messagebox.showerror("Invalid Input", "Please choose between 1 and 8 AI opponents.")
            return

        try:
            starting_stack = int(self.starting_stack_var.get())
        except (TypeError, ValueError, tk.TclError):
            messagebox.showerror("Invalid Input", "Please choose a valid starting stack amount.")
            return

        self._create_game(ai_count, starting_stack)
        self.setup_game_gui()
        self.play_hand()

    # ------------------------------------------------------------------
    # Game bootstrapping helpers
    # ------------------------------------------------------------------
    def _create_game(self, ai_count: int, starting_stack: int) -> None:
        """Create the underlying engine with the requested number of AIs."""

        total_players = ai_count + 1
        # Human player is always index 0 with no AI strategy assigned
        player_strategies = [None]

        for _ in range(ai_count):
            ai_strategy = _gui.RandomAIStrategy()
            player_strategies.append(ai_strategy)

        self.human_player_index = 0

        self.game = _gui.TexasHoldem(total_players, starting_stack, player_strategies)
        self.game_engine = self.game

        self.human_strategy = GUIHumanStrategy(self)
        # Replace the placeholder ``None`` with the GUI-aware human strategy.
        self.game.player_strategies[self.human_player_index] = self.human_strategy

    def setup_game_gui(self):  # pragma: no cover - GUI wiring
        """Build the standard table layout then add Change Players control."""

        super().setup_game_gui()
        self.selection_frame = None

        control_frame = tk.Frame(self.root, bg="gray")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            control_frame,
            text="Change Players",
            command=self.show_player_selection,
            font=("Arial", 12),
            bg="teal",
            fg="white",
        ).pack(side=tk.LEFT, padx=5)

    # Legacy helper used in tests; reuse the new bootstrapping logic
    def start_game_with_players(self, ai_count: int, starting_stack: int):
        _gui.tk = tk
        self._create_game(ai_count, starting_stack)
        self.setup_game_gui()
        self.play_hand()


GUIHumanStrategy = _gui.GUIHumanStrategy

__all__ = ["PokerGameGUI", "GUIHumanStrategy", "tk"]
