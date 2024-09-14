# texas_holdem.py

import random
import json
import os
from treys import Evaluator, Card  # Ensure treys is installed: pip install treys
from datetime import datetime


class TexasHoldemRules:
    def __init__(self, num_players=2, starting_stack=10000):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.deck = self._create_deck()
        self.hands = [[] for _ in range(num_players)]
        self.community_cards = []
        self.pot = 0
        self.bets = [0] * num_players
        self.player_chips = [starting_stack] * num_players
        self.active_players = [True] * num_players
        self.small_blind = 10
        self.big_blind = 20
        self.current_bet = 0
        self.previous_raise_amount = 0
        self.dealer_button = 0
        self.betting_history = []  # Stores actions per hand

    def _create_deck(self):
        suits = ['h', 'd', 'c', 's']  # h: hearts, d: diamonds, c: clubs, s: spades
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [rank + suit for suit in suits for rank in ranks]

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def deal(self):
        # Deal 2 cards to each active player
        for _ in range(2):
            for i in range(self.num_players):
                if self.active_players[i]:
                    self.hands[i].append(self.deck.pop(0))

    def post_blinds(self):
        small_blind_player = (self.dealer_button + 1) % self.num_players
        big_blind_player = (self.dealer_button + 2) % self.num_players

        # Small Blind
        self.player_chips[small_blind_player] -= self.small_blind
        self.bets[small_blind_player] = self.small_blind
        self.pot += self.small_blind
        self.betting_history.append(f"Player {small_blind_player + 1} posts small blind of {self.small_blind} chips.")
        print(f"Player {small_blind_player + 1} posts small blind of {self.small_blind} chips.")

        # Big Blind
        self.player_chips[big_blind_player] -= self.big_blind
        self.bets[big_blind_player] = self.big_blind
        self.pot += self.big_blind
        self.betting_history.append(f"Player {big_blind_player + 1} posts big blind of {self.big_blind} chips.")
        print(f"Player {big_blind_player + 1} posts big blind of {self.big_blind} chips.")

        self.current_bet = self.big_blind
        self.previous_raise_amount = self.big_blind
        self.current_player = (big_blind_player + 1) % self.num_players

    def bet(self, player_index, amount):
        bet_difference = amount - self.bets[player_index]
        if amount < self.current_bet:
            raise ValueError("Bet amount is less than the current bet.")
        if bet_difference > self.player_chips[player_index]:
            # Player goes all-in
            bet_difference = self.player_chips[player_index]
            amount = self.bets[player_index] + bet_difference
            self.player_chips[player_index] = 0
            self.betting_history.append(f"Player {player_index + 1} goes all-in with {bet_difference} chips.")
            print(f"Player {player_index + 1} goes all-in with {bet_difference} chips.")
        else:
            self.player_chips[player_index] -= bet_difference
            self.betting_history.append(f"Player {player_index + 1} raises to {amount} chips.")
            print(f"Player {player_index + 1} raises to {amount} chips.")

        self.pot += bet_difference
        self.bets[player_index] = amount

        if amount > self.current_bet:
            self.previous_raise_amount = amount - self.current_bet
            self.current_bet = amount
            self.betting_history.append(f"Current bet is now {self.current_bet} chips.")
            print(f"Current bet is now {self.current_bet} chips.")

    def advance_turn(self):
        self.current_player = (self.current_player + 1) % self.num_players
        # Skip inactive players or players with 0 chips
        for _ in range(self.num_players):
            if self.active_players[self.current_player] and self.player_chips[self.current_player] > 0:
                break
            self.current_player = (self.current_player + 1) % self.num_players

    def reset_bets(self):
        self.bets = [0] * self.num_players
        self.current_bet = 0
        self.previous_raise_amount = 0

    def deal_community_cards(self, round_stage):
        if round_stage == 'flop':
            # Burn a card
            burned = self.deck.pop(0)
            self.betting_history.append(f"Burned a card: {self._format_card(burned)}.")
            print(f"Burned a card: {self._format_card(burned)}.")
            # Deal the flop (3 cards)
            for _ in range(3):
                card = self.deck.pop(0)
                self.community_cards.append(card)
                self.betting_history.append(f"Dealt community card: {self._format_card(card)}.")
                print(f"Dealt community card: {self._format_card(card)}.")
        elif round_stage == 'turn':
            # Burn a card
            burned = self.deck.pop(0)
            self.betting_history.append(f"Burned a card: {self._format_card(burned)}.")
            print(f"Burned a card: {self._format_card(burned)}.")
            # Deal the turn (1 card)
            card = self.deck.pop(0)
            self.community_cards.append(card)
            self.betting_history.append(f"Dealt community card: {self._format_card(card)}.")
            print(f"Dealt community card: {self._format_card(card)}.")
        elif round_stage == 'river':
            # Burn a card
            burned = self.deck.pop(0)
            self.betting_history.append(f"Burned a card: {self._format_card(burned)}.")
            print(f"Burned a card: {self._format_card(burned)}.")
            # Deal the river (1 card)
            card = self.deck.pop(0)
            self.community_cards.append(card)
            self.betting_history.append(f"Dealt community card: {self._format_card(card)}.")
            print(f"Dealt community card: {self._format_card(card)}.")
        else:
            raise ValueError("Invalid round stage.")

    def _format_card(self, card):
        rank = card[0]
        suit = card[1]
        suit_symbols = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
        return f"{rank}{suit_symbols.get(suit, suit)}"


class TexasHoldem:
    def __init__(self, num_players, starting_stack, player_strategies):
        self.rules = TexasHoldemRules(num_players, starting_stack)
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.player_strategies = player_strategies  # List of strategy instances
        self.end_game_early = False
        self.winner = None
        self.hand_count = 0
        self.historical_actions = []
        self.history_limit = 500  # Save history every 500 hands

        # Ensure data directory exists
        if not os.path.exists('data'):
            os.makedirs('data')

    def initialize_game(self):
        self.rules.shuffle_deck()
        self.rules.post_blinds()
        self.deal_hands()
        self.hand_count += 1
        self.historical_actions.append({
            'hand_number': self.hand_count,
            'dealer': self.rules.dealer_button + 1,
            'actions': self.rules.betting_history.copy(),
            'community_cards': self.format_hand_display(self.rules.community_cards),
            'pot': self.rules.pot,
            'players': self.get_player_status()
        })
        self.rules.betting_history.clear()

    def deal_hands(self):
        self.rules.deal()

    def play_stage(self, stage):
        self.rules.deal_community_cards(stage)

    def process_action(self, player_index, action, raise_amount=None):
        try:
            if action == 'call':
                amount_to_call = self.rules.current_bet - self.rules.bets[player_index]
                if amount_to_call > self.rules.player_chips[player_index]:
                    amount_to_call = self.rules.player_chips[player_index]  # All-in
                    action_description = f"Player {player_index + 1} calls all-in with {amount_to_call} chips."
                    self.rules.betting_history.append(action_description)
                    print(action_description)
                else:
                    action_description = f"Player {player_index + 1} calls {amount_to_call} chips."
                    self.rules.betting_history.append(action_description)
                    print(action_description)
                new_bet = self.rules.bets[player_index] + amount_to_call
                self.rules.bet(player_index, new_bet)
            elif action in ['raise', 'bet']:
                if raise_amount is None:
                    print("Raise amount not provided. Defaulting to minimum raise.")
                    raise_amount = self.get_min_raise_amount(player_index)
                min_raise = self.get_min_raise_amount(player_index)
                amount_to_call = self.rules.current_bet - self.rules.bets[player_index]
                total_required = amount_to_call + raise_amount
                if raise_amount < min_raise:
                    print(f"Raise amount must be at least {min_raise} chips. Adjusting to minimum raise.")
                    action_description = f"Player {player_index + 1} attempted to raise {raise_amount} chips but minimum is {min_raise}. Adjusting raise."
                    self.rules.betting_history.append(action_description)
                    print(action_description)
                    raise_amount = min_raise
                    total_required = amount_to_call + raise_amount
                if total_required > self.rules.player_chips[player_index]:
                    # Player goes all-in
                    total_required = self.rules.player_chips[player_index]
                    action_description = f"Player {player_index + 1} does not have enough chips to raise {raise_amount} chips. Going all-in with {total_required} chips."
                    self.rules.betting_history.append(action_description)
                    print(action_description)
                new_bet = self.rules.bets[player_index] + total_required
                self.rules.bet(player_index, new_bet)
            elif action == 'fold':
                self.rules.active_players[player_index] = False
                action_description = f"Player {player_index + 1} folds."
                self.rules.betting_history.append(action_description)
                print(action_description)
                active_players = [i for i in range(self.num_players) if self.rules.active_players[i]]
                if len(active_players) == 1:
                    self.end_game_early = True
                    self.winner = active_players[0]
            elif action == 'check':
                action_description = f"Player {player_index + 1} checks."
                self.rules.betting_history.append(action_description)
                print(action_description)
                if self.rules.current_bet != self.rules.bets[player_index]:
                    print("Cannot check when there is a bet to call. Defaulting to call.")
                    self.process_action(player_index, 'call')
            else:
                action_description = f"Player {player_index + 1} made an invalid action '{action}' and is folding by default."
                self.rules.betting_history.append(action_description)
                print(action_description)
                self.process_action(player_index, 'fold')
        except ValueError as ve:
            print(f"Error processing action: {ve}. Adjusting action.")
            self.rules.betting_history.append(f"Error processing action for Player {player_index + 1}: {ve}. Adjusting to call.")
            print(f"Error processing action for Player {player_index + 1}: {ve}. Adjusting to call.")
            # Decide on a fallback action, such as folding or calling
            if action in ['raise', 'bet']:
                # Default to calling if raise was invalid
                self.process_action(player_index, 'call')
            else:
                # Default to folding for other invalid actions
                self.process_action(player_index, 'fold')

    def betting_round(self):
        if self.rules.current_bet == 0:
            self.rules.reset_bets()
        self.rules.current_player = (self.rules.dealer_button + 1) % self.num_players
        first_player = True
        while True:
            all_bets_equal = all(
                (self.rules.bets[i] == self.rules.current_bet or
                 not self.rules.active_players[i] or
                 self.rules.player_chips[i] == 0)
                for i in range(self.num_players)
            )
            if all_bets_equal and not first_player:
                break
            first_player = False

            player_index = self.rules.current_player
            if self.rules.active_players[player_index] and self.rules.player_chips[player_index] > 0:
                strategy = self.player_strategies[player_index]
                action, raise_amount = strategy.choose_action(self, player_index)
                self.process_action(player_index, action, raise_amount)
                if self.end_game_early:
                    break
            self.rules.advance_turn()

    def get_min_raise_amount(self, player_index):
        if self.rules.previous_raise_amount:
            min_raise = self.rules.previous_raise_amount
        else:
            min_raise = self.rules.big_blind
        return min_raise

    def get_max_raise_amount(self, player_index):
        amount_to_call = self.rules.current_bet - self.rules.bets[player_index]
        max_raise = self.rules.player_chips[player_index] - amount_to_call
        return max_raise

    def perform_showdown(self):
        active_players = [i for i in range(self.num_players) if self.rules.active_players[i]]
        if not active_players:
            return None  # No active players

        if len(active_players) == 1:
            # Only one player remains; they are the winner
            return active_players[0], None  # No need to evaluate hands

        # Evaluate hands
        evaluator = Evaluator()
        # Convert community cards to treys format
        community_cards = [Card.new(card) for card in self.rules.community_cards]

        player_scores = {}
        player_best_hands = {}
        for player_index in active_players:
            player_hand = self.rules.hands[player_index]
            # Convert player's hole cards to treys format
            hole_cards = [Card.new(card) for card in player_hand]
            score = evaluator.evaluate(community_cards, hole_cards)
            best_hand = evaluator.get_rank_class(score)
            player_scores[player_index] = score
            player_best_hands[player_index] = best_hand

        # Determine the winner(s)
        best_score = min(player_scores.values())
        winners = [player for player, score in player_scores.items() if score == best_score]

        if len(winners) == 1:
            winner = winners[0]
        else:
            # Handle ties (split pot)
            winner = winners  # List of winners

        return winner, player_best_hands

    def reset_for_next_hand(self):
        # Reset all player chips to starting stack
        self.rules.player_chips = [self.rules.starting_stack] * self.num_players
        self.rules.active_players = [True] * self.num_players
        self.rules.dealer_button = (self.rules.dealer_button + 1) % self.num_players

        # Reset game state
        self.rules.deck = self.rules._create_deck()
        self.rules.shuffle_deck()
        self.rules.hands = [[] for _ in range(self.num_players)]
        self.rules.community_cards = []
        self.rules.pot = 0
        self.rules.bets = [0] * self.num_players
        self.rules.current_player = (self.rules.dealer_button + 1) % self.num_players
        self.rules.current_bet = 0
        self.rules.previous_raise_amount = 0
        self.rules.betting_history = []
        self.end_game_early = False
        self.winner = None

    def play_game(self):
        print(f"--- Hand {self.hand_count + 1} ---")
        self.initialize_game()
        self.show_player_hands()
        self.display_player_chips()  # Display chips before pre-flop

        # Pre-flop betting round
        print("\n--- Pre-flop Betting Round ---")
        self.betting_round()
        if self.end_game_early:
            self.declare_winner()
            self.save_history_if_needed()
            return

        stages = ['flop', 'turn', 'river']
        for stage in stages:
            self.play_stage(stage)
            self.print_community_cards(stage)
            self.display_player_chips()  # Display chips before betting
            print(f"\n--- Betting Round after the {stage.capitalize()} ---")
            self.betting_round()
            if self.end_game_early:
                self.declare_winner()
                self.save_history_if_needed()
                return

        # Showdown
        print("\n--- Showdown ---")
        for i in range(self.num_players):
            if self.rules.active_players[i]:
                hand_str = self.format_hand_display(self.rules.hands[i])
                print(f"Player {i + 1}'s hand: {hand_str}")

        winner, player_best_hands = self.perform_showdown()
        if winner is not None:
            self.show_winner(winner, player_best_hands)
        else:
            print("\nNo winner could be determined.")

        self.save_history_if_needed()

    def declare_winner(self):
        print(f"\n--- Hand Conclusion ---")
        print(f"All other players have folded.")
        print(f"The winner is Player {self.winner + 1}!")
        print(f"Pot won: {self.rules.pot} chips.")
        self.rules.player_chips[self.winner] += self.rules.pot
        self.historical_actions[-1]['winner'] = f"Player {self.winner + 1}"
        self.historical_actions[-1]['pot_won'] = self.rules.pot

    def show_winner(self, winner, player_best_hands):
        evaluator = Evaluator()
        hand_rankings = {}
        for player_index in player_best_hands:
            rank_class = player_best_hands[player_index]
            hand_name = evaluator.class_to_string(rank_class)
            hand_rankings[player_index] = hand_name

        if isinstance(winner, list):
            print("\nIt's a tie between the following players:")
            for w in winner:
                print(f"Player {w + 1} with a {hand_rankings[w]}")
            print(f"Pot split between players.")
            split_pot = self.rules.pot // len(winner)
            for w in winner:
                self.rules.player_chips[w] += split_pot
            self.historical_actions[-1]['winner'] = [f"Player {w + 1}" for w in winner]
            self.historical_actions[-1]['pot_won'] = self.rules.pot
        else:
            print(f"\nThe winner is Player {winner + 1} with a {hand_rankings[winner]}!")
            print(f"Pot won: {self.rules.pot} chips.")
            self.rules.player_chips[winner] += self.rules.pot
            self.historical_actions[-1]['winner'] = f"Player {winner + 1}"
            self.historical_actions[-1]['pot_won'] = self.rules.pot

    def format_hand_display(self, hand):
        # Mapping from treys-compatible suits to display-friendly symbols
        suit_symbols = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
        return ' '.join([card[0] + suit_symbols.get(card[1], card[1]) for card in hand])

    def display_player_chips(self):
        print("\n--- Player Chip Counts ---")
        for i in range(self.num_players):
            status = "Active" if self.rules.active_players[i] else "Eliminated"
            print(f"Player {i + 1}: {self.rules.player_chips[i]} chips ({status})")
        print("---------------------------\n")

    def show_player_hands(self):
        for i in range(self.num_players):
            strategy = self.player_strategies[i]
            if strategy.is_human:
                hand_str = self.format_hand_display(self.rules.hands[i])
                print(f"Player {i + 1}'s hand: {hand_str}")
            else:
                pass  # AI hands can be hidden or shown as desired

    def print_community_cards(self, stage):
        community_str = self.format_hand_display(self.rules.community_cards)
        print(f"\nCommunity cards after the {stage.capitalize()}: {community_str}")
        print(f"Pot: {self.rules.pot} chips.")

    def get_player_status(self):
        return {
            f"Player {i + 1}": {
                "chips": self.rules.player_chips[i],
                "status": "Active" if self.rules.active_players[i] else "Eliminated"
            }
            for i in range(self.num_players)
        }

    def save_history_if_needed(self):
        if self.hand_count % self.history_limit == 0:
            timestamp = datetime.now().strftime("%y%m%d%H%M%S")
            filename = f"data/historical_actions_{timestamp}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(self.historical_actions, f, indent=4)
                print(f"\n--- Historical data saved to {filename} ---")
                self.historical_actions.clear()
            except Exception as e:
                print(f"Error saving historical data: {e}")
