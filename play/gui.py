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
            try:                self.background_image = tk.PhotoImage(file=background_image_path)
            except tk.TclError as e:
                print(f"Error loading background image: {e}")
        
        self.card_images = {}
        self._load_card_images()
        
        # Game Engine Setup (will be initialized after player selection)
        self.human_player_index = 0
        self.game_engine = None  # Will be created after player selection
        self.ai_count = 1  # Default, will be set by user choice

        # Initialize game state variables (will be updated by engine)
        self.player_hand = []
        self.community_cards = []
        self.pot = 0
        self.player_money = 1000  # Default starting amount
        self.ai_money = 1000  # Default, will be updated        # Define fonts and colors
        try:
            self.font_title = font.Font(family="Arial", size=16, weight="bold")
            self.font_label = font.Font(family="Arial", size=12)
            self.font_button = font.Font(family="Arial", size=12, weight="bold")
        except Exception:
            class _DummyFont:
                def __init__(self):
                    pass
            self.font_title = self.font_label = self.font_button = _DummyFont()
        
        self.color_background = "#006400"  # Dark Green (felt color if no image)
        self.color_frame_bg = "#004D00"    # Slightly darker green for frames
        self.color_text = "#FFFFFF"        # White
        self.color_button_bg = "#8B0000"   # Dark Red
        self.color_button_fg = "#FFFFFF"   # White

        # GUI elements
        self.setup_gui()
        
        # Show player selection dialog before starting the game
        self.show_player_selection()
        
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
             main_container.configure(bg="") # This might not make it fully transparent depending on TK version / OS        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # AI Info Frame (Top)
        ai_frame = tk.Frame(main_container, bg=self.color_frame_bg, pady=10)
        ai_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Create container for AI player displays
        self.ai_players_frame = tk.Frame(ai_frame, bg=self.color_frame_bg)
        self.ai_players_frame.pack()
        
        # AI money labels will be created dynamically based on number of AI players
        self.ai_money_labels = []
        
        # Initialize AI display if game engine exists
        if hasattr(self, 'game_engine') and self.game_engine:
            self._setup_ai_display()

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
        # Place check button, maybe adjust grid columns or add a new row        self.check_button.grid(row=0, column=3, sticky="ew", padx=5) # Added check button

        # Button frame for next hand / quit options (initially empty)
        self.button_frame = tk.Frame(main_container, bg=self.color_background)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
    def start_game(self):
        if not self.game_engine:
            print("Game engine not initialized. Please select number of players first.")
            return
            
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
        
        self.update_display() # This will redraw cards, update money labels etc.
        self._update_ai_display() # Update all AI player displays
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
                txt_label = tk.Label(self.player_hand_frame, text=card_str, relief=tk.RIDGE, bg="lightgrey", fg="black", font=self.font_label)
                txt_label.pack(side=tk.LEFT, padx=5, pady=2)
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
                txt_label = tk.Label(self.community_cards_frame, text=card_str, relief=tk.RIDGE, bg="lightgrey", fg="black", font=self.font_label)
                txt_label.pack(side=tk.LEFT, padx=2)
                print(f"Image not found for community card {card_str} (key: {image_key})")

        self.pot_label.config(text=f"Pot: ${self.pot}")
        self.player_money_label.config(text=f"Your Money: ${self.player_money}")
        # AI money is now handled by _update_ai_display() method

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
            # If current_bet is X, and player's current bet is Y, and player wants total bet to be Z (bet_amount from entry):
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
        """Handles AI turns, game stage progression, and schedules next step."""
        rules = self.game_engine.rules

        if self.game_engine.end_game_early:
            self._handle_hand_end()
            return

        if rules.betting_round_is_over():
            print(f"Betting round over. Actions this round: {rules.actions_this_round}, Pot: {rules.pot}")
            rules.end_betting_round_cleanup()  # Resets bets, current_bet, actions_this_round

            current_stage_index = -1
            if not rules.community_cards: current_stage_index = 0  # Pre-flop was done
            elif len(rules.community_cards) == 3: current_stage_index = 1  # Flop was done
            elif len(rules.community_cards) == 4: current_stage_index = 2  # Turn was done
            elif len(rules.community_cards) == 5:  # River was done
                self._handle_hand_end()  # Showdown
                return

            next_stages = ['flop', 'turn', 'river']
            if current_stage_index < len(next_stages):
                next_stage_name = next_stages[current_stage_index]
                print(f"Dealing {next_stage_name}...")
                self.game_engine.play_stage(next_stage_name)
                
                # Determine starting player for the new round (SB or first active player after button)
                rules.current_player = (rules.dealer_button + 1) % rules.num_players
                for _ in range(rules.num_players):
                    if rules.active_players[rules.current_player] and rules.player_chips[rules.current_player] > 0:
                        break
                    rules.current_player = (rules.current_player + 1) % rules.num_players

                rules.actions_this_round = 0 # Reset for new betting round
                # Blinds are posted at initialize_game, not between rounds like flop/turn/river.
                # Small/big blind players might need to act again if they just posted.
                # The current_player logic above should handle finding the first to act.
            else:  # Should go to showdown
                self._handle_hand_end()
                return

        self._sync_gui_with_engine_state()  # Update GUI before AI or human turn

        if rules.current_player == self.human_player_index:
            # Check if human player is still active and has chips
            if not rules.active_players[self.human_player_index] or rules.player_chips[self.human_player_index] == 0:
                # Human is out or folded, treat as AI turn to advance game
                print(f"Human player (P{self.human_player_index+1}) is out or folded. Advancing.")
                rules.advance_turn() # This will find the next active player
                self.root.after(50, self._handle_game_progression) # Schedule next progression
                return

            print(f"Human player's (P{self.human_player_index+1}) turn.")
            self._update_action_buttons_state()
            return  # Wait for human action, do not reschedule automatically

        # AI's turn
        # Ensure AI player is active and has chips
        current_ai_player = rules.current_player
        if not rules.active_players[current_ai_player] or rules.player_chips[current_ai_player] == 0:
            print(f"AI player (P{current_ai_player+1}) is out or folded. Advancing.")
            rules.advance_turn()
            self.root.after(50, self._handle_game_progression)
            return

        if self.game_engine.player_strategies[current_ai_player]:
            print(f"AI player's (P{current_ai_player+1}) turn.")
            self._update_action_buttons_state()  # Disable human buttons during AI turn

            ai_strategy = self.game_engine.player_strategies[current_ai_player]
            action, amount = ai_strategy.choose_action(self.game_engine, current_ai_player)
            print(f"AI (P{current_ai_player+1}) chose: {action}, amount: {amount}")
            
            try:
                self.game_engine.process_action(current_ai_player, action, raise_amount=amount if action == 'bet' or action == 'raise' else None)
                rules.actions_this_round += 1
            except ValueError as e:
                print(f"Error processing AI (P{current_ai_player+1}) action {action} {amount}: {e}")
                # If AI makes an invalid move, it should ideally be handled by the strategy or engine.
                # For now, we might just advance turn to prevent getting stuck.
                # A better solution would be for AI strategy to always return valid moves.
                # Or for process_action to have a fallback (e.g. AI checks/folds if action is invalid)
                print(f"AI (P{current_ai_player+1}) made an invalid move. Forcing check/fold or advancing.")
                # Simplest: just advance. This could be problematic if AI always errors.
                # A more robust AI would not error here.
                # For now, let's assume AI action is valid or process_action handles errors gracefully (e.g. auto-fold)
                # If process_action raises ValueError, the game might get stuck on this AI.
                # A simple recovery: if AI action fails, try to make it check or fold.
                try:
                    if self.game_engine.rules.current_bet > self.game_engine.rules.bets[current_ai_player]: # Must call or fold
                        self.game_engine.process_action(current_ai_player, 'fold')
                    else: # Can check
                        self.game_engine.process_action(current_ai_player, 'check')
                    rules.actions_this_round += 1
                except Exception as e_fallback:
                    print(f"AI (P{current_ai_player+1}) fallback action failed: {e_fallback}. Advancing turn.")
                    rules.advance_turn() # Last resort to prevent infinite loop on faulty AI

            self._sync_gui_with_engine_state()  # Update after AI action
            self.root.after(50, self._handle_game_progression) # Schedule next progression
        else:
            # This case should ideally not be reached if human player is handled and AI strategies are present
            print(f"No strategy for current player {current_ai_player}, advancing turn.")
            rules.advance_turn()
            self.root.after(50, self._handle_game_progression) # Schedule next progression


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
        
        # Show a "Next Hand" button instead of automatically starting a new game
        self._show_next_hand_option()
    
    def _show_next_hand_option(self):
        """Show option to start next hand instead of automatically starting."""
        # Clear action buttons
        for widget in self.button_frame.winfo_children():
            widget.destroy()
            
        # Add "Next Hand" button
        next_hand_btn = tk.Button(
            self.button_frame, 
            text="Start Next Hand", 
            command=self._start_next_hand,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            width=15
        )
        next_hand_btn.pack(side=tk.LEFT, padx=10)
        
        # Add "Change Players" button
        change_players_btn = tk.Button(
            self.button_frame,
            text="Change Players",
            command=self.show_player_selection,
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white", 
            width=15
        )
        change_players_btn.pack(side=tk.LEFT, padx=10)
        
        # Add "Quit Game" button
        quit_btn = tk.Button(
            self.button_frame,
            text="Quit Game",
            command=self.root.quit,
            font=("Arial", 12, "bold"), 
            bg="#f44336",
            fg="white",
            width=15
        )
        quit_btn.pack(side=tk.LEFT, padx=10)
    
    def _start_next_hand(self):
        """Start the next hand when user clicks the button."""
        self.start_game()

    def show_player_selection(self):
        """Show dialog to select number of AI players"""
        # Hide the main game interface temporarily
        for widget in self.root.winfo_children():
            widget.pack_forget()
        
        # Create player selection frame
        selection_frame = tk.Frame(self.root, bg=self.color_background)
        selection_frame.pack(expand=True, fill=tk.BOTH)
        
        # Title
        title_label = tk.Label(
            selection_frame, 
            text="Texas Hold'em Poker Setup", 
            font=("Arial", 24, "bold"),
            bg=self.color_background, 
            fg=self.color_text        )
        title_label.pack(pady=50)
        
        # Instructions
        instructions = tk.Label(
            selection_frame,
            text="Choose how many AI opponents you want to play against:",
            font=("Arial", 14),
            bg=self.color_background,
            fg=self.color_text
        )
        instructions.pack(pady=20)
        
        # Player selection frame
        input_frame = tk.Frame(selection_frame, bg=self.color_background)
        input_frame.pack(pady=30)
        
        # Store selection frame reference for cleanup
        self.selection_frame = selection_frame
        
        # AI count input frame
        count_frame = tk.Frame(input_frame, bg=self.color_background)
        count_frame.pack(pady=20)
        
        tk.Label(
            count_frame,
            text="Number of AI opponents:",
            font=("Arial", 12),
            bg=self.color_background,
            fg=self.color_text
        ).pack(side=tk.LEFT, padx=10)
        
        # Spinbox for selecting number of AI players (1-8 for reasonable game size)
        self.ai_count_var = tk.StringVar(value="1")
        ai_spinbox = tk.Spinbox(
            count_frame,
            from_=1,
            to=8,
            textvariable=self.ai_count_var,
            font=("Arial", 12),
            width=5,
            justify=tk.CENTER
        )
        ai_spinbox.pack(side=tk.LEFT, padx=10)
        
        # Start game button
        start_button = tk.Button(
            input_frame,
            text="Start Game",
            font=("Arial", 14, "bold"),
            bg=self.color_button_bg,
            fg=self.color_button_fg,            width=15,
            height=2,
            command=self.start_game_from_selection
        )
        start_button.pack(pady=20)
    
    def start_game_from_selection(self):
        """Get AI count from spinbox and start game"""
        try:
            ai_count = int(self.ai_count_var.get())
            if ai_count < 1 or ai_count > 8:
                messagebox.showerror("Invalid Input", "Please select between 1 and 8 AI opponents.")
                return
            self.start_game_with_players(ai_count)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of AI opponents.")
    
    def start_game_with_players(self, ai_count):
        """Initialize game with selected number of AI players"""
        total_players = ai_count + 1  # AI players + 1 human player
        
        # Create AI strategies for each AI player
        ai_strategy = PlaceholderAIStrategy()
        player_strategies = [None]  # Human player (index 0) has no strategy
        
        # Add AI strategies for each AI player
        for i in range(ai_count):
            player_strategies.append(ai_strategy)
        
        # Create new game engine with selected number of players
        self.game_engine = TexasHoldem(
            num_players=total_players, 
            starting_stack=1000, 
            player_strategies=player_strategies
        )
        
        # Update player info
        self.human_player_index = 0
        self.ai_count = ai_count
        self.player_money = self.game_engine.rules.player_chips[self.human_player_index]
          # Store AI money for display (we'll show total AI money or individual later)
        self.ai_money = sum(self.game_engine.rules.player_chips[1:]) if ai_count > 0 else 0
        
        # Clean up selection interface
        self.selection_frame.destroy()
        
        # Show the main game interface that was hidden
        self.setup_game_gui()
          # Start the actual game
        self.start_game()
    
    def setup_game_gui(self):
        """Setup the main game GUI after player selection"""
        # The main game widgets were hidden with pack_forget() during player selection
        # We need to make them visible again
        
        # Find and restore the main container widget
        for widget in self.root.winfo_children():
            # Look for the main container frame (not the selection frame)
            if isinstance(widget, tk.Frame) and widget != getattr(self, 'selection_frame', None):
                widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                break
        
        # If main container wasn't found, recreate the GUI
        if not any(isinstance(widget, tk.Frame) and widget.winfo_manager() == 'pack' 
                  for widget in self.root.winfo_children()):
            self.setup_gui()
        
        # Setup AI display now that game engine exists
        self._setup_ai_display()
        
    def _setup_ai_display(self):
        """Setup AI player display based on number of AI players"""
        # Clear existing AI labels
        for label in self.ai_money_labels:
            label.destroy()
        self.ai_money_labels = []
        
        if not self.game_engine:
            return
            
        # Create labels for each AI player
        for i in range(1, self.game_engine.num_players):  # Skip player 0 (human)
            ai_player_num = i
            ai_money = self.game_engine.rules.player_chips[i]
            
            label_text = f"AI Player {ai_player_num}: ${ai_money}"
            ai_label = tk.Label(
                self.ai_players_frame, 
                text=label_text, 
                font=self.font_label, 
                bg=self.color_frame_bg, 
                fg=self.color_text
            )
            ai_label.pack(side=tk.LEFT, padx=20)
            self.ai_money_labels.append(ai_label)
    
    def _update_ai_display(self):
        """Update AI player money display"""
        if not self.game_engine or not self.ai_money_labels:
            return
            
        for i, label in enumerate(self.ai_money_labels):
            ai_index = i + 1  # AI players start at index 1
            if ai_index < len(self.game_engine.rules.player_chips):
                ai_money = self.game_engine.rules.player_chips[ai_index]
                label.config(text=f"AI Player {ai_index}: ${ai_money}")

if __name__ == "__main__":
    PokerGameGUI(None)

