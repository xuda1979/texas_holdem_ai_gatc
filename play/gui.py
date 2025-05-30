import tkinter as tk
from tkinter import messagebox, font
import os
import sys

# Add project root to sys.path to allow imports from game_engine and play
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game_engine.texas_holdem import TexasHoldem, TexasHoldemRules # Assuming TexasHoldemRules might be useful for type hints or constants
from play.strategies import PlaceholderAIStrategy, HumanStrategy

class PokerGameGUI:
    def __init__(self, trainer=None): # trainer is optional now
        self.root = tk.Tk()
        self.root.title("Texas Hold'em Poker")
        self.root.geometry("1000x700")

        # Load background image
        self.background_image = None
        background_image_path = "play/assets/poker_table_background.png"
        if os.path.exists(background_image_path):
            try:
                self.background_image = tk.PhotoImage(file=background_image_path)
            except tk.TclError as e:
                print(f"Error loading background image: {e}")
        
        self.card_images = {}
        self._load_card_images()
        
        # Game Engine Setup
        self.human_player_index = 0
        ai_strategy = PlaceholderAIStrategy()
        # Human strategy is None as GUI handles human actions.
        player_strategies = [None, ai_strategy] 
        self.game_engine = TexasHoldem(num_players=2, starting_stack=1000, player_strategies=player_strategies)

        # Initialize game state variables (will be updated by engine)
        self.player_hand = []
        self.community_cards = []
        self.pot = 0
        self.player_money = self.game_engine.rules.player_chips[self.human_player_index]
        self.ai_money = self.game_engine.rules.player_chips[1] if self.game_engine.num_players > 1 else 0


        # Define fonts and colors
        self.font_title = font.Font(family="Arial", size=16, weight="bold")
        self.font_label = font.Font(family="Arial", size=12)
        self.font_button = font.Font(family="Arial", size=12, weight="bold")
        
        self.color_background = "#006400"  # Dark Green (felt color if no image)
        self.color_frame_bg = "#004D00"    # Slightly darker green for frames
        self.color_text = "#FFFFFF"        # White
        self.color_button_bg = "#8B0000"   # Dark Red
        self.color_button_fg = "#FFFFFF"   # White

        # GUI elements
        self.setup_gui()
        
        # Start the game
        self.start_game()
        
        self.root.mainloop()

    def _load_card_images(self):
        card_image_dir = "play/card_images"
        available_files = []
        if os.path.exists(card_image_dir):
            available_files = [f for f in os.listdir(card_image_dir) if f.endswith(".png")]
        
        for filename in available_files:
            card_key = filename.split('.')[0] 
            try:
                image_path = os.path.join(card_image_dir, filename)
                # Resize images if they are too large, e.g., to 1/5th of original if they are from Byron Knoll set
                img = tk.PhotoImage(file=image_path)
                self.card_images[card_key] = img.subsample(5, 5) # Adjust subsample rate as needed
            except tk.TclError as e:
                print(f"Error loading image {filename}: {e}")
                self.card_images[card_key] = None

    def _get_card_image_key(self, card_string):
        """Converts card string (e.g., 'A♠', 'K♦', or 'As', 'Kd') to image key (e.g., 'As', 'Kd')."""
        if not card_string or len(card_string) < 2:
            return None
        if card_string == "New Card" or card_string == "Card": # Handle placeholder community cards
            return "placeholder"


        # Handle "A♠" style format
        if len(card_string) == 2 and card_string[1] in ['♠', '♥', '♦', '♣']:
            rank = card_string[0]
            suit_char = card_string[1]
            suit_map = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
            suit = suit_map.get(suit_char)
            if not suit: return None
            if rank == '1' and card_string.startswith('10'): rank = 'T' # e.g. "10♠"
            elif rank in ['A', 'K', 'Q', 'J', 'T']: pass # Added 'T' here
            elif rank.isdigit() and '2' <= rank <= '9': pass
            else: # This case should ideally not be reached if ranks are A,K,Q,J,T,2-9
                  # or for "10" if it's passed as two chars like "10s" (handled below)
                  # However, if a string like "10♠" is passed, the initial rank is '1'.
                  # The first condition handles "10♠" correctly.
                  # This 'else' might catch other unexpected formats.
                return None
            return f"{rank}{suit}"
        # Handle "As" style format
        elif len(card_string) == 2 and card_string[1] in ['s', 'h', 'd', 'c']:
            rank = card_string[0]
            if rank in ['A', 'K', 'Q', 'J', 'T'] or (rank.isdigit() and '2' <= rank <= '9'):
                return card_string
        return None

    def setup_gui(self):
        # Background (if image loaded)
        if self.background_image:
            self.background_label = tk.Label(self.root, image=self.background_image)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
            # Configure root background for areas not covered by image (though relwidth/height should cover)
            self.root.configure(bg=self.color_background)
        else:
            self.root.configure(bg=self.color_background)

        # Main container frame to put other frames on top of background label
        main_container = tk.Frame(self.root, bg=self.root.cget('bg')) # transparent if possible
        if self.background_image: # If background image exists, make frame transparent to it
             main_container.configure(bg="") # This might not make it fully transparent depending on TK version / OS
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


        # AI Info Frame (Top)
        ai_frame = tk.Frame(main_container, bg=self.color_frame_bg, pady=10)
        ai_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.ai_money_label = tk.Label(ai_frame, text=f"AI Money: ${self.ai_money}", font=self.font_label, bg=self.color_frame_bg, fg=self.color_text)
        self.ai_money_label.pack()
        # Placeholder for AI hand/status if needed later

        # Community Cards & Pot Frame (Middle)
        community_pot_frame = tk.Frame(main_container, bg=self.color_frame_bg, pady=10)
        community_pot_frame.pack(side=tk.TOP, fill=tk.X, pady=5, expand=True)
        
        tk.Label(community_pot_frame, text="Community Cards:", font=self.font_title, bg=self.color_frame_bg, fg=self.color_text).pack()
        self.community_cards_frame = tk.Frame(community_pot_frame, bg=self.color_frame_bg, pady=5)
        self.community_cards_frame.pack()

        self.pot_label = tk.Label(community_pot_frame, text=f"Pot: ${self.pot}", font=self.font_label, bg=self.color_frame_bg, fg=self.color_text)
        self.pot_label.pack(pady=5)

        # Player Info & Hand Frame (Bottom)
        player_frame = tk.Frame(main_container, bg=self.color_frame_bg, pady=10)
        player_frame.pack(side=tk.TOP, fill=tk.X, pady=5) # Changed from BOTTOM to TOP to stack below community
        
        self.player_money_label = tk.Label(player_frame, text=f"Your Money: ${self.player_money}", font=self.font_label, bg=self.color_frame_bg, fg=self.color_text)
        self.player_money_label.pack()
        
        tk.Label(player_frame, text="Your Hand:", font=self.font_title, bg=self.color_frame_bg, fg=self.color_text).pack()
        self.player_hand_frame = tk.Frame(player_frame, bg=self.color_frame_bg, pady=5)
        self.player_hand_frame.pack()
        
        # Action Buttons Frame (Very Bottom)
        actions_frame = tk.Frame(main_container, bg=self.color_background, pady=10) # Match main background or distinct
        actions_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Bet input sub-frame
        bet_input_frame = tk.Frame(actions_frame, bg=self.color_background)
        bet_input_frame.pack(pady=5) # Add some padding below it, before buttons

        self.bet_amount_label = tk.Label(bet_input_frame, text="Bet Amount:", font=self.font_label, bg=self.color_background, fg=self.color_text)
        self.bet_amount_label.pack(side=tk.LEFT, padx=5)
        self.bet_entry = tk.Entry(bet_input_frame, font=self.font_label, width=10)
        self.bet_entry.pack(side=tk.LEFT)

        # Action buttons sub-frame (to keep them grouped)
        action_buttons_frame = tk.Frame(actions_frame, bg=self.color_background)
        action_buttons_frame.pack(pady=5)
        action_buttons_frame.grid_columnconfigure(0, weight=1) 
        action_buttons_frame.grid_columnconfigure(1, weight=1)
        action_buttons_frame.grid_columnconfigure(2, weight=1)

        button_options = {'font': self.font_button, 'bg': self.color_button_bg, 'fg': self.color_button_fg, 'padx': 10, 'pady': 5}
        self.bet_button = tk.Button(action_buttons_frame, text="Bet", command=self.player_bet, **button_options)
        self.bet_button.grid(row=0, column=0, sticky="ew", padx=5)

        self.call_button = tk.Button(action_buttons_frame, text="Call", command=self.player_call, **button_options)
        self.call_button.grid(row=0, column=1, sticky="ew", padx=5)
        
        self.fold_button = tk.Button(action_buttons_frame, text="Fold", command=self.player_fold, **button_options)
        self.fold_button.grid(row=0, column=2, sticky="ew", padx=5)
        
        self.check_button = tk.Button(action_buttons_frame, text="Check", command=self.player_check, **button_options)
        # Place check button, maybe adjust grid columns or add a new row
        self.check_button.grid(row=0, column=3, sticky="ew", padx=5) # Added check button


    def start_game(self):
        self.game_engine.initialize_game() # Deals new hand, posts blinds
        self._sync_gui_with_engine_state()
        self._handle_game_progression() # Check for AI turn or next stage

    def _sync_gui_with_engine_state(self):
        """Updates GUI elements based on the current game engine state."""
        rules = self.game_engine.rules
        self.player_hand = rules.hands[self.human_player_index]
        self.community_cards = rules.community_cards
        self.pot = rules.pot
        self.player_money = rules.player_chips[self.human_player_index]
        if self.game_engine.num_players > 1:
            self.ai_money = rules.player_chips[1]
        
        self.update_display() # This will redraw cards, update money labels etc.
        self._update_action_buttons_state()

    def _update_action_buttons_state(self):
        """Enable/Disable action buttons based on game state and current player."""
        is_human_turn = (self.game_engine.rules.current_player == self.human_player_index) and \
                        not self.game_engine.end_game_early and \
                        len(self.game_engine.rules.community_cards) < 5 # Simple check for hand not over
        
        # Basic enable/disable all buttons based on turn
        for btn in [self.bet_button, self.call_button, self.fold_button, self.check_button]:
            btn.config(state=tk.NORMAL if is_human_turn else tk.DISABLED)

        if is_human_turn:
            # More specific logic:
            # Can check? (current bet is 0 or player's current bet matches table's current bet)
            can_check = (self.game_engine.rules.current_bet == self.game_engine.rules.bets[self.human_player_index])
            self.check_button.config(state=tk.NORMAL if can_check else tk.DISABLED)
            self.call_button.config(state=tk.NORMAL if not can_check else tk.DISABLED)
            # Bet button should be available if check is available (to open betting) or if raise is possible
            # Fold is always available on player's turn
            # For simplicity, this basic enable/disable is okay for now.

    def update_display(self):
        # Clear previous card images from their specific frames
        for widget in self.player_hand_frame.winfo_children():
            widget.destroy()
        
        for widget in self.community_cards_frame.winfo_children():
            widget.destroy()

        # Display player hand
        for card_str in self.player_hand:
            image_key = self._get_card_image_key(card_str)
            if image_key and image_key in self.card_images and self.card_images[image_key]:
                img_label = tk.Label(self.player_hand_frame, image=self.card_images[image_key], bg=self.color_frame_bg)
                img_label.pack(side=tk.LEFT, padx=2)
            else:
                # Fallback to text
                txt_label = tk.Label(self.player_hand_frame, text=card_str, relief=tk.RIDGE, padding=5, bg="lightgrey", fg="black", font=self.font_label)
                txt_label.pack(side=tk.LEFT, padx=2)
                if image_key != "placeholder": # Avoid printing for "New Card" if it's handled as placeholder
                    print(f"Image not found for player card {card_str} (key: {image_key})")

        # Display community cards
        for card_str in self.community_cards:
            image_key = self._get_card_image_key(card_str)
            if image_key and image_key in self.card_images and self.card_images[image_key]:
                img_label = tk.Label(self.community_cards_frame, image=self.card_images[image_key], bg=self.color_frame_bg)
                img_label.pack(side=tk.LEFT, padx=2)
            elif image_key == "placeholder": # Handle placeholder text for unrevealed cards
                 txt_label = tk.Label(self.community_cards_frame, text="?", relief=tk.RIDGE, width=4, height=2, bg="grey", fg="white", font=self.font_title) # Placeholder style
                 txt_label.pack(side=tk.LEFT, padx=2)
            else:
                # Fallback to text
                txt_label = tk.Label(self.community_cards_frame, text=card_str, relief=tk.RIDGE, padding=5, bg="lightgrey", fg="black", font=self.font_label)
                txt_label.pack(side=tk.LEFT, padx=2)
                print(f"Image not found for community card {card_str} (key: {image_key})")

        self.pot_label.config(text=f"Pot: ${self.pot}")
        self.player_money_label.config(text=f"Your Money: ${self.player_money}")
        self.ai_money_label.config(text=f"AI Money: ${self.ai_money}")

    def _handle_player_action(self, action_type, amount=0):
        if self.game_engine.rules.current_player != self.human_player_index:
            messagebox.showwarning("Not your turn", "It's not your turn to act.")
            return
        
        try:
            print(f"Human (Player {self.human_player_index + 1}) action: {action_type}, amount: {amount}")
            self.game_engine.process_action(self.human_player_index, action_type, raise_amount=amount if action_type == 'bet' or action_type == 'raise' else None)
            self._sync_gui_with_engine_state()
            self._handle_game_progression()
        except ValueError as e:
            messagebox.showerror("Action Error", str(e))

    def player_bet(self):
        try:
            bet_amount_str = self.bet_entry.get()
            if not bet_amount_str:
                messagebox.showerror("Invalid Bet", "Please enter a bet amount.")
                return
            bet_amount = int(bet_amount_str)

            if bet_amount <= 0:
                messagebox.showerror("Invalid Bet", "Bet amount must be positive.")
                return
            
            # Note: game_engine.process_action will validate if player has enough money 
            # and if the bet/raise is valid against current_bet and previous_raise_amount.
            # The amount here for 'bet' in process_action is the raise_amount *on top of* any call.
            # However, for a simple GUI bet button, user enters the TOTAL they want their bet to be.
            # Let's adjust to send the "raise amount" part or adapt process_action.
            # For now, we assume player_bet means "raise to this total amount" or "open bet with this amount"
            # This needs careful alignment with engine's process_action.
            # A common GUI approach: if current_bet > 0, this is a raise. If current_bet == 0, this is an open bet.
            
            # Let's assume bet_amount is the total bet the player wants to make for this round.
            # The engine's process_action for 'bet' takes 'raise_amount'.
            # If current bet is X, and player's current bet is Y, and player wants total bet to be Z (bet_amount from entry):
            # Amount to call = X - Y
            # Additional raise amount = Z - X (if Z > X)
            # So, raise_amount for process_action should be Z - X.
            
            current_table_bet = self.game_engine.rules.current_bet
            player_current_bet_in_round = self.game_engine.rules.bets[self.human_player_index]
            
            if bet_amount < current_table_bet + self.game_engine.get_min_raise_amount(self.human_player_index) and current_table_bet > 0 : # Must raise at least min_raise if raising
                 if bet_amount > player_current_bet_in_round and bet_amount < current_table_bet : # Trying to bet less than current bet but more than their own
                      messagebox.showerror("Invalid Bet", f"Cannot bet {bet_amount}. Must call {current_table_bet} or raise to at least {current_table_bet + self.game_engine.get_min_raise_amount(self.human_player_index)}")
                      return
            
            # For 'bet' action type, process_action expects raise_amount
            # If it's an opening bet, current_table_bet is 0. raise_amount = bet_amount.
            # If it's a raise, raise_amount is bet_amount - current_table_bet.
            # This is still a bit complex. Let's simplify: GUI 'Bet' button means player wants to bet/raise.
            # The amount entered is the *total* new bet. process_action should ideally handle this.
            # The current process_action for 'bet' takes 'raise_amount' which is the additional part.
            # For now, let's pass bet_amount as if it's the 'raise_amount' on top of a call.
            # This implies the player has to calculate the "additional" part. This is not user friendly.
            
            # Simpler: if action is 'bet', it's treated as 'raise' in the engine.
            # The engine's `process_action` for 'raise'/'bet' expects `raise_amount` to be the *additional* amount.
            # Let's make GUI's "Bet" button mean "raise by this amount" or "bet this amount if opening"
            
            # If current bet is 50, player has 10 in pot, entry is 100.
            # Call is 40. Raise by 100 means total new bet is 10 (current) + 40 (call) + 100 (raise) = 150.
            # This is complex. For now, `bet_amount` from entry is the `raise_amount` for `process_action`.
            
            self._handle_player_action('bet', bet_amount) # 'bet' implies raise or open bet
            self.bet_entry.delete(0, tk.END)

        except ValueError:
            messagebox.showerror("Invalid Bet", "Please enter a valid number for the bet amount.")
            self.bet_entry.delete(0, tk.END)
        except Exception as e: # Catch other game logic errors from process_action
            messagebox.showerror("Action Error", str(e))
            self.bet_entry.delete(0, tk.END)


    def player_fold(self):
        self._handle_player_action('fold')

    def player_call(self):
        self._handle_player_action('call')
        
    def player_check(self):
        self._handle_player_action('check')

    def _handle_game_progression(self):
        """Handles AI turns and game stage progression."""
        rules = self.game_engine.rules
        while True:
            if self.game_engine.end_game_early:
                self._handle_hand_end()
                return

            if rules.betting_round_is_over():
                print(f"Betting round over. Actions this round: {rules.actions_this_round}, Pot: {rules.pot}")
                rules.end_betting_round_cleanup() # Resets bets, current_bet, actions_this_round
                
                current_stage_index = -1
                if not rules.community_cards: current_stage_index = 0 # Pre-flop was done
                elif len(rules.community_cards) == 3: current_stage_index = 1 # Flop was done
                elif len(rules.community_cards) == 4: current_stage_index = 2 # Turn was done
                elif len(rules.community_cards) == 5: # River was done
                    self._handle_hand_end() # Showdown
                    return

                next_stages = ['flop', 'turn', 'river']
                if current_stage_index < len(next_stages):
                    next_stage_name = next_stages[current_stage_index]
                    print(f"Dealing {next_stage_name}...")
                    self.game_engine.play_stage(next_stage_name)
                    rules.current_player = (rules.dealer_button + 1) % rules.num_players # SB acts first post-flop
                    # Skip inactive players
                    for _ in range(rules.num_players):
                        if rules.active_players[rules.current_player] and rules.player_chips[rules.current_player] > 0:
                            break
                        rules.current_player = (rules.current_player + 1) % rules.num_players
                    rules.actions_this_round = 0 # Reset for new betting round
                else: # Should go to showdown
                    self._handle_hand_end()
                    return
            
            self._sync_gui_with_engine_state() # Update GUI before AI or human turn

            if rules.current_player == self.human_player_index:
                print(f"Human player's (P{self.human_player_index+1}) turn.")
                self._update_action_buttons_state()
                return # Wait for human action

            # AI's turn
            if self.game_engine.player_strategies[rules.current_player]:
                print(f"AI player's (P{rules.current_player+1}) turn.")
                self._update_action_buttons_state() # Disable buttons during AI turn
                # self.root.update_idletasks() # Ensure GUI updates if AI is slow

                ai_strategy = self.game_engine.player_strategies[rules.current_player]
                action, amount = ai_strategy.choose_action(self.game_engine, rules.current_player)
                print(f"AI (P{rules.current_player+1}) chose: {action}, amount: {amount}")
                self.game_engine.process_action(rules.current_player, action, raise_amount=amount if action == 'bet' or action == 'raise' else None)
                rules.actions_this_round += 1
                self._sync_gui_with_engine_state() # Update after AI action
            else: # Should not happen if human player is handled
                rules.advance_turn()


    def _handle_hand_end(self):
        """Handles the end of a hand (showdown or all but one folded)."""
        rules = self.game_engine.rules
        winner_message = ""

        if self.game_engine.end_game_early: # Someone folded
            winner = self.game_engine.winner
            winner_message = f"Player {winner + 1} wins the pot of ${rules.pot} as others folded!"
            rules.player_chips[winner] += rules.pot
        else: # Showdown
            print("Performing showdown...")
            # Ensure all players' hands are visible for showdown if they are active
            # (Already handled by update_display if hands are correctly assigned)
            showdown_winner, player_best_hands = self.game_engine.perform_showdown()
            if isinstance(showdown_winner, list): # Tie
                player_hands_str = "\n".join([f"Player {idx+1}: {self.game_engine.format_hand_display(rules.hands[idx])} ({player_best_hands[idx]})" for idx in showdown_winner])
                winner_message = f"Split pot! Winners: {', '.join([f'Player {w+1}' for w in showdown_winner])}\n{player_hands_str}\nPot: ${rules.pot}"
                split_amount = rules.pot // len(showdown_winner)
                for w_idx in showdown_winner:
                    rules.player_chips[w_idx] += split_amount
            elif showdown_winner is not None:
                hand_str = self.game_engine.format_hand_display(rules.hands[showdown_winner])
                winner_message = f"Player {showdown_winner + 1} wins with {player_best_hands[showdown_winner]} ({hand_str})!\nPot: ${rules.pot}"
                rules.player_chips[showdown_winner] += rules.pot
            else:
                winner_message = "Error in determining winner at showdown."
        
        messagebox.showinfo("Hand Over", winner_message)
        rules.pot = 0 
        self._sync_gui_with_engine_state() # Update money after pot distribution
        
        # For now, we can just start a new game. Later, add a "New Hand" button.
        # messagebox.showinfo("Next Hand", "Starting new hand.")
        self.start_game()


if __name__ == "__main__":
    PokerGameGUI(None)

