import tkinter as tk
from tkinter import messagebox, simpledialog

# Pillow is optional: the GUI falls back to text-based cards when it is not
# installed.  Import lazily so tests can run in minimal environments.
try:  # pragma: no cover - used only when Pillow is available
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover - Pillow missing
    Image = ImageTk = None
import os
import sys

# Add project root to sys.path to allow imports from game_engine and play
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game_engine.texas_holdem import TexasHoldem

from poker_ai.gui.playStrategy import HumanStrategy, RandomAIStrategy


class GUIHumanStrategy(HumanStrategy):
    """GUI version of HumanStrategy that uses GUI callbacks instead of console input."""

    def __init__(self, gui_instance):
        self.gui_instance = gui_instance
        self.pending_action = None
        self.action_complete = False

    def choose_action(self, game, player_index):
        """Request action from GUI and wait for response."""
        self.action_complete = False
        self.pending_action = None

        # Calculate action info
        amount_to_call = game.rules.current_bet - game.rules.bets[player_index]

        # Request action from GUI
        self.gui_instance.request_human_action(game, player_index, amount_to_call)

        # Wait for GUI to provide action (this will be set by GUI button callbacks)
        while not self.action_complete:
            self.gui_instance.root.update()  # Process GUI events

        return self.pending_action

    def set_action(self, action, amount=None):
        """Called by GUI to set the human player's action."""
        self.pending_action = (action, amount)
        self.action_complete = True


class PokerGameGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Texas Hold'em Poker - Human vs AI")
        self.root.geometry("1400x900")

        # Attributes expected by legacy tests
        self.human_player_index = 0
        self.game_engine = TexasHoldem
          # Card image cache
        self.card_images = {}
        self.card_back_image = None
        self._load_card_images()

        # Game state
        self.game = None
        self.human_strategy = None
        self.current_game_state = {}

        # GUI elements
        self.info_frame = None
        self.cards_frame = None
        self.actions_frame = None
        self.status_label = None

        # Delegate GUI setup and start to helper methods so tests can easily
        # patch them.
        self.setup_gui()
        self.start_game()
        # Start the Tk main loop immediately
        self.root.mainloop()

    def _get_card_image_key(self, card: str | None) -> str | None:
        """Convert a card string to the image key used in the GUI."""
        if not card:
            return None

        placeholder_cards = {"New Card", "Card"}
        if card in placeholder_cards:
            return "placeholder"

        if len(card) < 2:
            return None

        rank = card[0]
        suit = card[1]
        suit_map = {
            "♠": "s",
            "♣": "c",
            "♥": "h",
            "♦": "d",
            "s": "s",
            "c": "c",
            "h": "h",
            "d": "d",
        }

        if suit not in suit_map:
            return None

        return f"{rank}{suit_map[suit]}"

    def _handle_player_action(self, action: str, amount: int | None = None):
        """Process an action for the human player."""
        engine = getattr(self, 'game_engine', None)
        if not engine or not hasattr(engine, 'rules') or not hasattr(engine.rules, 'current_player'):
            messagebox.showwarning("Game Error", "No active game found.")
            return

        # For GUI, human is always player 0
        if engine.rules.current_player != 0:
            messagebox.showwarning("Not your turn", "It's not your turn to act.")
            return

        # Process the action through the human strategy
        if self.human_strategy:
            self.human_strategy.set_action(action, amount)

        if hasattr(engine, 'process_action'):
            engine.process_action(0, action, raise_amount=amount)
            self._sync_gui_with_engine_state()
            self._handle_game_progression()

    def _sync_gui_with_engine_state(self):
        """Synchronize cached state values with the game engine."""
        engine = getattr(self, 'game_engine', None)
        if not engine:
            return
        rules = engine.rules

        # Cache relevant state for tests or GUI refreshes
        self.player_hand = rules.hands[0] if hasattr(rules, 'hands') else []
        self.community_cards = getattr(rules, 'community_cards', [])
        self.pot = getattr(rules, 'pot', 0)
        self.player_money = rules.player_chips[0] if hasattr(rules, 'player_chips') else 0
        self.ai_money = (
            rules.player_chips[1] if hasattr(rules, 'player_chips') and len(rules.player_chips) > 1 else 0
        )

        # Update display
        self.update_display()
        self._update_action_buttons_state()

    # Placeholder helpers referenced in tests
    def _handle_game_progression(self):
        pass

    def _update_action_buttons_state(self):
        pass

    def setup_gui(self):
        """Hook for setting up the initial GUI layout."""
        self.setup_initial_gui()

    def start_game(self):
        """Hook for additional start behaviour."""
        pass

    def _load_card_images(self):
        """Load all card images from the card_images directory."""
        card_images_dir = os.path.join(os.path.dirname(__file__), "card_images")

        # Standard card size for display (larger for better visibility)
        card_width, card_height = 100, 140

        if not os.path.exists(card_images_dir):
            print(f"Warning: Card images directory not found at {card_images_dir}")
            return

        try:
            # Load all card images
            ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            suits = ['c', 'd', 'h', 's']

            for rank in ranks:
                for suit in suits:
                    card_name = f"{rank}{suit}"
                    image_path = os.path.join(card_images_dir, f"{card_name}.png")

                    if os.path.exists(image_path):
                        # Load and resize image
                        img = Image.open(image_path)
                        img = img.resize((card_width, card_height), Image.Resampling.LANCZOS)
                        self.card_images[card_name] = ImageTk.PhotoImage(img)
                    else:
                        print(f"Warning: Card image not found: {image_path}")
                          # Create a placeholder card back (simple rectangle)
            back_img = Image.new('RGB', (card_width, card_height), color='darkblue')
            self.card_back_image = ImageTk.PhotoImage(back_img)

            print(f"Loaded {len(self.card_images)} card images")

        except ImportError:
            print("Warning: PIL (Pillow) not available. Card images will not be displayed.")
            print("Install Pillow with: pip install Pillow")
        except Exception as e:
            print(f"Error loading card images: {e}")
            print("Falling back to text display.")

    # Backwards compatibility for tests expecting load_card_images
    def load_card_images(self):
        self._load_card_images()

    def create_card_label(self, parent, card_name, show_back=False):
        """Create a label with a card image."""
        if show_back or card_name not in self.card_images:
            # Show card back or placeholder
            if self.card_back_image:
                return tk.Label(parent, image=self.card_back_image, bg=parent.cget('bg'),
                               relief="solid", bd=1)
            else:
                # Fallback to text if no images available
                return tk.Label(parent, text=card_name if not show_back else "??",
                               font=("Arial", 10, "bold"), bg="white", fg="black",
                               width=8, height=6, relief="raised", bd=2)
        else:
            # Show actual card image
            return tk.Label(parent, image=self.card_images[card_name], bg=parent.cget('bg'),
                           relief="solid", bd=1)

    def setup_initial_gui(self):
        """Setup the initial game configuration screen."""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Title
        title_label = tk.Label(
            self.root,
            text="Texas Hold'em Poker - Human vs AI",
            font=("Arial", 24, "bold"),
            bg="green",
            fg="white"
        )
        title_label.pack(pady=20, fill=tk.X)

        # Configuration frame
        config_frame = tk.Frame(self.root, bg="darkgreen")
        config_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Total players
        tk.Label(
            config_frame,
            text="Total Players (2-10):",
            font=("Arial", 14),
            bg="darkgreen",
            fg="white"
        ).pack(pady=10)

        self.total_players_var = tk.StringVar(value="3")
        total_players_spinbox = tk.Spinbox(
            config_frame,
            from_=2,
            to=10,
            textvariable=self.total_players_var,
            font=("Arial", 12),
            width=5
        )
        total_players_spinbox.pack(pady=5)

        # Number of humans (always 1 for this implementation)
        tk.Label(
            config_frame,
            text="Human Players: 1 (You)",
            font=("Arial", 14),
            bg="darkgreen",
            fg="white"
        ).pack(pady=10)

        # Starting stack
        tk.Label(
            config_frame,
            text="Starting Stack:",
            font=("Arial", 14),
            bg="darkgreen",
            fg="white"
        ).pack(pady=10)

        self.starting_stack_var = tk.StringVar(value="10000")
        stack_options = ["1000", "5000", "10000", "20000", "50000"]
        stack_combo = tk.OptionMenu(config_frame, self.starting_stack_var, *stack_options)
        stack_combo.config(font=("Arial", 12))
        stack_combo.pack(pady=5)

        # Start game button
        start_button = tk.Button(
            config_frame,
            text="Start Game",
            font=("Arial", 16, "bold"),
            bg="red",
            fg="white",
            command=self.start_new_game,
            width=15,
            height=2
        )
        start_button.pack(pady=30)

    def start_new_game(self):
        """Start a new game with the configured settings."""
        try:
            total_players = int(self.total_players_var.get())
            starting_stack = int(self.starting_stack_var.get())

            if not (2 <= total_players <= 10):
                messagebox.showerror("Invalid Input", "Total players must be between 2 and 10.")
                return

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return

        # Create player strategies - always 1 human + (total-1) AIs
        player_strategies = []
          # Create GUI human strategy
        self.human_strategy = GUIHumanStrategy(self)
        player_strategies.append(self.human_strategy)

        # Add AI strategies
        num_ai = total_players - 1
        for _ in range(num_ai):
            player_strategies.append(RandomAIStrategy())

        # Create game
        self.game = TexasHoldem(total_players, starting_stack, player_strategies)

        # Setup game GUI
        self.setup_game_gui()

        # Start the game loop
        self.play_hand()

    # Backwards compatibility for older tests
    def start_game_with_players(self, ai_count: int, starting_stack: int):
        """Legacy helper used by tests to start a game with a given number of AIs."""
        total_players = ai_count + 1
        player_strategies = [GUIHumanStrategy(self)]
        for _ in range(ai_count):
            player_strategies.append(RandomAIStrategy())
        self.game = TexasHoldem(total_players, starting_stack, player_strategies)
        self.setup_game_gui()
        self.play_hand()

    def setup_game_gui(self):
        """Setup the main game interface."""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Game Starting...",
            font=("Arial", 14),
            bg="navy",
            fg="white",
            height=2
        )
        self.status_label.pack(fill=tk.X)

        # Game info frame
        self.info_frame = tk.Frame(self.root, bg="darkgreen", height=150)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.info_frame.pack_propagate(False)
          # Community cards and pot frame (increased height for card images)
        self.cards_frame = tk.Frame(self.root, bg="green", height=250)
        self.cards_frame.pack(fill=tk.X, padx=10, pady=5)
        self.cards_frame.pack_propagate(False)

        # Human player info frame (increased height for card images)
        self.player_frame = tk.Frame(self.root, bg="darkblue", height=200)
        self.player_frame.pack(fill=tk.X, padx=10, pady=5)
        self.player_frame.pack_propagate(False)

        # Actions frame
        self.actions_frame = tk.Frame(self.root, bg="red", height=100)
        self.actions_frame.pack(fill=tk.X, padx=10, pady=5)
        self.actions_frame.pack_propagate(False)

        # Control buttons frame
        control_frame = tk.Frame(self.root, bg="gray")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            control_frame,
            text="New Game",
            command=self.setup_initial_gui,
            font=("Arial", 12),
            bg="orange",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="Quit",
            command=self.root.quit,
            font=("Arial", 12),
            bg="darkred",
            fg="white"
        ).pack(side=tk.RIGHT, padx=5)

    def update_display(self):
        """Update the GUI display with current game state."""
        if not self.game:
            return

        # Gracefully handle tests that bypass GUI setup
        if not self.info_frame or not self.cards_frame or not self.player_frame:
            return

        # Clear frames
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        for widget in self.cards_frame.winfo_children():
            widget.destroy()
        for widget in self.player_frame.winfo_children():
            widget.destroy()

        # Update info frame - show AI players
        tk.Label(
            self.info_frame,
            text="AI Players:",
            font=("Arial", 14, "bold"),
            bg="darkgreen",
            fg="white"
        ).pack()

        ai_info = ""
        for i in range(1, self.game.num_players):  # Skip player 0 (human)
            ai_info += f"AI Player {i+1}: ${self.game.rules.player_chips[i]} chips  "

        tk.Label(
            self.info_frame,
            text=ai_info,
            font=("Arial", 12),
            bg="darkgreen",
            fg="white"
        ).pack()
          # Update cards frame - Community Cards with images
        community_label = tk.Label(
            self.cards_frame,
            text="Community Cards:",
            font=("Arial", 14, "bold"),
            bg="green",
            fg="white"
        )
        community_label.pack(pady=5)

        # Community cards frame for images
        community_cards_frame = tk.Frame(self.cards_frame, bg="green")
        community_cards_frame.pack(pady=5)

        if self.game.rules.community_cards:
            for card in self.game.rules.community_cards:
                card_label = self.create_card_label(community_cards_frame, card)
                card_label.pack(side=tk.LEFT, padx=5, pady=5)
        else:
            tk.Label(
                community_cards_frame,
                text="No community cards yet",
                font=("Arial", 12),
                bg="green",
                fg="white"
            ).pack()

        tk.Label(
            self.cards_frame,
            text=f"Pot: ${self.game.rules.pot}",
            font=("Arial", 16, "bold"),
            bg="green",
            fg="yellow"
        ).pack(pady=5)

        # Update player frame - human player (always index 0) with card images
        player_hand_label = tk.Label(
            self.player_frame,
            text="Your Hand:",
            font=("Arial", 14, "bold"),
            bg="darkblue",
            fg="white"
        )
        player_hand_label.pack(pady=5)

        # Player cards frame for images
        player_cards_frame = tk.Frame(self.player_frame, bg="darkblue")
        player_cards_frame.pack(pady=5)

        if self.game.rules.hands[0]:
            for card in self.game.rules.hands[0]:
                card_label = self.create_card_label(player_cards_frame, card)
                card_label.pack(side=tk.LEFT, padx=5, pady=5)
        else:
            tk.Label(
                player_cards_frame,
                text="No cards yet",
                font=("Arial", 12),
                bg="darkblue",
                fg="white"
            ).pack()

        tk.Label(
            self.player_frame,
            text=f"Your Chips: ${self.game.rules.player_chips[0]}",
            font=("Arial", 14),
            bg="darkblue",
            fg="white"
        ).pack()

        # Show current bet info
        amount_to_call = self.game.rules.current_bet - self.game.rules.bets[0]
        if amount_to_call > 0:
            call_text = f"Amount to call: ${amount_to_call}"
        else:
            call_text = "You can check"

        tk.Label(
            self.player_frame,
            text=call_text,
            font=("Arial", 12),
            bg="darkblue",
            fg="yellow"
        ).pack()

    def request_human_action(self, game, player_index, amount_to_call):
        """Called by GUIHumanStrategy when human action is needed."""
        # Update display first
        self.update_display()

        # Clear actions frame
        for widget in self.actions_frame.winfo_children():
            widget.destroy()

        # Update status
        self.status_label.config(text=f"Your turn! Pot: ${game.rules.pot} | Your chips: ${game.rules.player_chips[0]}")

        # Create action buttons based on game state
        tk.Label(
            self.actions_frame,
            text="Choose your action:",
            font=("Arial", 14, "bold"),
            bg="red",
            fg="white"
        ).pack()

        button_frame = tk.Frame(self.actions_frame, bg="red")
        button_frame.pack()

        if amount_to_call > 0:
            # Can call, raise, or fold
            tk.Button(
                button_frame,
                text=f"Call ${amount_to_call}",
                command=lambda: self.human_action('call'),
                font=("Arial", 12, "bold"),
                bg="green",
                fg="white",
                width=12
            ).pack(side=tk.LEFT, padx=5)

            tk.Button(
                button_frame,
                text="Raise",
                command=lambda: self.human_action('raise'),
                font=("Arial", 12, "bold"),
                bg="orange",
                fg="white",
                width=12
            ).pack(side=tk.LEFT, padx=5)

            tk.Button(
                button_frame,
                text="Fold",
                command=lambda: self.human_action('fold'),
                font=("Arial", 12, "bold"),
                bg="darkred",
                fg="white",
                width=12
            ).pack(side=tk.LEFT, padx=5)
        else:
            # Can check or bet
            tk.Button(
                button_frame,
                text="Check",
                command=lambda: self.human_action('check'),
                font=("Arial", 12, "bold"),
                bg="blue",
                fg="white",
                width=12
            ).pack(side=tk.LEFT, padx=5)

            tk.Button(
                button_frame,
                text="Bet",
                command=lambda: self.human_action('bet'),
                font=("Arial", 12, "bold"),
                bg="orange",
                fg="white",
                width=12
            ).pack(side=tk.LEFT, padx=5)

    def human_action(self, action):
        """Handle human player action."""
        if action in ['raise', 'bet']:
            # Get raise/bet amount
            min_raise = self.game.get_min_raise_amount(0)  # Human is always player 0
            max_raise = self.game.get_max_raise_amount(0)

            if max_raise < min_raise:
                amount_to_call = self.game.rules.current_bet - self.game.rules.bets[0]
                if amount_to_call > 0:
                    messagebox.showwarning("Cannot Raise", "You don't have enough chips to raise. You can only call or fold.")
                else:
                    messagebox.showwarning("Cannot Bet", "You don't have enough chips to bet. You can only check.")
                return

            amount = simpledialog.askinteger(
                "Raise/Bet Amount",
                f"Enter {action} amount (min: ${min_raise}, max: ${max_raise}):",
                minvalue=min_raise,
                maxvalue=max_raise
            )

            if amount is None:  # User cancelled
                return

            self.human_strategy.set_action(action, amount)
        else:
            self.human_strategy.set_action(action, None)

    def play_hand(self):
        """Play a single hand of poker."""
        if not self.game:
            return

        self.status_label.config(text="Playing hand...")
        self.update_display()

        try:
            # This will call the human strategy when it's the human's turn
            self.game.play_game()

            # Hand completed
            self.status_label.config(text="Hand completed! Click 'Next Hand' to continue.")

            # Clear actions frame and show next hand button
            for widget in self.actions_frame.winfo_children():
                widget.destroy()

            tk.Label(
                self.actions_frame,
                text="Hand Completed!",
                font=("Arial", 16, "bold"),
                bg="red",
                fg="white"
            ).pack()

            next_hand_btn = tk.Button(
                self.actions_frame,
                text="Next Hand",
                command=self.next_hand,
                font=("Arial", 14, "bold"),
                bg="green",
                fg="white",
                width=15
            )
            next_hand_btn.pack(pady=10)

        except Exception as e:
            messagebox.showerror("Game Error", f"An error occurred: {str(e)}")
            self.setup_initial_gui()

    def next_hand(self):
        """Start the next hand."""
        if self.game:
            self.game.reset_for_next_hand()
            self.play_hand()

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    gui = PokerGameGUI()
    gui.run()

