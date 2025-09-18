"""Simplified Texas Hold'em poker engine used for training."""

# Ruff checks are suppressed for this module because it relies on dynamic
# typing and has several long logging lines.
# ruff: noqa

import json
import logging
import os
import random
from typing import Optional

from gatc_holdem.engine.rules import min_raise_to
from gatc_poker.pots import compute_side_pots as gatc_compute_side_pots

# ``treys`` provides fast poker hand evaluation but is optional in our test
# environment.  To keep the engine lightweight, we attempt to import the real
# dependency and fall back to minimal stub implementations when it is absent.
#
# The stub only exposes the small surface area exercised by the unit tests:
# ``Card.new`` for converting a string like ``'Ah'`` and ``Evaluator`` methods
# ``evaluate`` and ``get_rank_class``/``class_to_string``.  The implementation
# simply returns constant values, which is sufficient because the tests never
# rely on actual hand strengths—they merely ensure that the engine can be
# instantiated without the third‑party package.
try:  # pragma: no cover - exercised implicitly when treys is installed
    from treys import Card, Evaluator  # type: ignore
except Exception:  # pragma: no cover - treys missing

    class _Evaluator:  # minimal stub
        def evaluate(self, community_cards, hole_cards):
            return 0

        def get_rank_class(self, score):
            return 0

        def class_to_string(self, rank_class):
            return "High Card"

    class _Card:  # minimal stub
        @staticmethod
        def new(card_str):
            return card_str

    Evaluator = _Evaluator
    Card = _Card


class CardDeck(list):
    """Simple list subclass exposing a ``cards`` attribute for tests."""

    @property
    def cards(self):
        return list(self)


SUITS = ["h", "d", "c", "s"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
from datetime import datetime


class TexasHoldemRules:
    def __init__(self, num_players=2, starting_stack=10000, verbose=True):
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
        self.actions_this_round = 0  # Counter for actions in the current betting round
        self.last_raiser = None  # Tracks the player index of the last raiser in a betting round
        self.total_bets_this_hand = [
            0
        ] * num_players  # Tracks total bets per player for the current hand
        self.betting_round = "pre-flop"  # Add this line
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            self.logger.info(msg)

    def _create_deck(self):
        suits = ["h", "d", "c", "s"]  # h: hearts, d: diamonds, c: clubs, s: spades
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        return CardDeck([rank + suit for suit in suits for rank in ranks])

    def shuffle_deck(self):
        # Ensure the deck is actually shuffled
        random.shuffle(self.deck)

    def rotate_dealer(self):
        """Move the dealer button to the next player."""
        self.dealer_button = (self.dealer_button + 1) % self.num_players

    def deal(self):
        # Deal 2 cards to each active player
        for _ in range(2):
            for i in range(self.num_players):
                if self.active_players[i]:
                    self.hands[i].append(self.deck.pop(0))

    def post_blinds(self):
        structured_blind_actions = []
        small_blind_player = (self.dealer_button + 1) % self.num_players
        big_blind_player = (self.dealer_button + 2) % self.num_players

        # Small Blind
        self.player_chips[small_blind_player] -= self.small_blind
        self.bets[small_blind_player] = self.small_blind
        self.pot += self.small_blind
        self.total_bets_this_hand[small_blind_player] += self.small_blind
        self.betting_history.append((str(small_blind_player), ("bet", self.small_blind)))
        self._log(f"Player {small_blind_player + 1} posts small blind of {self.small_blind} chips.")
        structured_blind_actions.append((str(small_blind_player), ("bet", self.small_blind)))

        # Big Blind
        self.player_chips[big_blind_player] -= self.big_blind
        self.bets[big_blind_player] = self.big_blind
        self.pot += self.big_blind
        self.total_bets_this_hand[big_blind_player] += self.big_blind
        self.betting_history.append((str(big_blind_player), ("bet", self.big_blind)))
        self._log(f"Player {big_blind_player + 1} posts big blind of {self.big_blind} chips.")
        structured_blind_actions.append((str(big_blind_player), ("bet", self.big_blind)))

        self.current_bet = self.big_blind
        self.previous_raise_amount = self.big_blind
        self.current_player = (big_blind_player + 1) % self.num_players
        return structured_blind_actions

    def bet(self, player_index, amount):
        bet_difference = amount - self.bets[player_index]
        if amount < self.current_bet:
            raise ValueError("Bet amount is less than the current bet.")
        if bet_difference > self.player_chips[player_index]:
            # Player goes all-in
            bet_difference = self.player_chips[player_index]
            amount = self.bets[player_index] + bet_difference
            self.player_chips[player_index] = 0
            self._log(f"Player {player_index + 1} goes all-in with {bet_difference} chips.")
        else:
            self.player_chips[player_index] -= bet_difference
            self._log(f"Player {player_index + 1} raises to {amount} chips.")

        self.pot += bet_difference
        self.total_bets_this_hand[player_index] += (
            bet_difference  # Accumulate total bet for the hand
        )
        self.bets[player_index] = amount  # This is total bet for the current round

        if amount > self.current_bet:
            self.previous_raise_amount = amount - self.current_bet
            self.current_bet = amount
            self._log(f"Current bet is now {self.current_bet} chips.")

    def advance_turn(self):
        self.current_player = (self.current_player + 1) % self.num_players
        # Skip inactive players or players with 0 chips
        for _ in range(self.num_players):
            if (
                self.active_players[self.current_player]
                and self.player_chips[self.current_player] > 0
            ):
                break
            self.current_player = (self.current_player + 1) % self.num_players

    def reset_bets(self):
        self.bets = [0] * self.num_players
        self.current_bet = 0
        self.previous_raise_amount = 0
        self.actions_this_round = 0  # Reset actions_this_round here as well

    def betting_round_is_over(self):
        """Return True if all active players have matched the current bet."""
        active_players = [
            i
            for i in range(self.num_players)
            if self.active_players[i] and self.player_chips[i] > 0
        ]
        if not active_players:
            return True
        all_settled = all(self.bets[i] == self.current_bet for i in active_players)
        return all_settled and self.actions_this_round >= len(active_players)

    # betting_round_is_over originally removed, but provided here for backward compatibility

    def end_betting_round_cleanup(self):
        """
        Resets betting state for the start of a new round (street).
        The pot itself is accumulated incrementally during bets.
        Calls self.reset_bets().
        """
        self.reset_bets()  # Resets self.bets, self.current_bet, self.previous_raise_amount, and self.actions_this_round
        # self.current_player for the next round will be set by the TexasHoldem.betting_round method.

    def deal_community_cards(self, round_stage):
        if round_stage == "flop":
            # Burn a card
            burned = self.deck.pop(0)
            self._log(f"Burned a card: {self._format_card(burned)}.")
            # Deal the flop (3 cards)
            for _ in range(3):
                card = self.deck.pop(0)
                self.community_cards.append(card)
                self._log(f"Dealt community card: {self._format_card(card)}.")
        elif round_stage == "turn":
            # Burn a card
            burned = self.deck.pop(0)
            self._log(f"Burned a card: {self._format_card(burned)}.")
            # Deal the turn (1 card)
            card = self.deck.pop(0)
            self.community_cards.append(card)
            self._log(f"Dealt community card: {self._format_card(card)}.")
        elif round_stage == "river":
            # Burn a card
            burned = self.deck.pop(0)
            self._log(f"Burned a card: {self._format_card(burned)}.")
            # Deal the river (1 card)
            card = self.deck.pop(0)
            self.community_cards.append(card)
            self._log(f"Dealt community card: {self._format_card(card)}.")
        else:
            raise ValueError("Invalid round stage.")

    def _format_card(self, card):
        rank = card[0]
        suit = card[1]
        suit_symbols = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
        return f"{rank}{suit_symbols.get(suit, suit)}"

    def clone(self) -> "TexasHoldemRules":
        """Return a lightweight copy of the mutable rule state."""

        clone = self.__class__.__new__(self.__class__)

        # Immutable or shared objects can be copied directly.
        clone.num_players = self.num_players
        clone.starting_stack = self.starting_stack
        clone.small_blind = self.small_blind
        clone.big_blind = self.big_blind
        clone.logger = self.logger
        clone.verbose = self.verbose
        clone.betting_round = self.betting_round
        clone.dealer_button = self.dealer_button
        clone.pot = self.pot
        clone.current_bet = self.current_bet
        clone.previous_raise_amount = self.previous_raise_amount
        clone.actions_this_round = self.actions_this_round
        clone.last_raiser = self.last_raiser

        # Lists are re-created to avoid accidental sharing during simulations.
        clone.deck = CardDeck(self.deck)
        clone.hands = [list(hand) for hand in self.hands]
        clone.community_cards = list(self.community_cards)
        clone.bets = list(self.bets)
        clone.player_chips = list(self.player_chips)
        clone.active_players = list(self.active_players)
        clone.betting_history = list(self.betting_history)
        clone.total_bets_this_hand = list(self.total_bets_this_hand)

        if hasattr(self, "current_player"):
            clone.current_player = self.current_player

        return clone


class MockPlayer:
    def __init__(self, player_id: str, hand: list, stack: int):
        self.player_id = player_id
        self.hand = hand
        self.stack = stack
        self.current_bet_in_round = 0


class TexasHoldem:
    def __init__(self, num_players, starting_stack=1000, player_strategies=None, verbose=True):
        self.rules = TexasHoldemRules(num_players, starting_stack, verbose=verbose)
        self.num_players = num_players
        self.starting_stack = starting_stack
        if player_strategies is None:
            player_strategies = [None] * num_players
        self.player_strategies = player_strategies  # List of strategy instances
        self.end_game_early = False
        self.winner = None
        self.last_winner = None
        self.hand_count = 0
        self.historical_actions = []
        self.history_limit = 500  # Save history every 500 hands
        self.current_hand_initial_actions = []
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self._pending_showdown_winnings: Optional[dict[int, int]] = None
        if not os.path.exists("data"):
            os.makedirs("data")

    def _log(self, msg: str) -> None:
        if self.verbose:
            self.logger.info(msg)

    def clone(self) -> "TexasHoldem":
        """Return a copy containing only the mutable state required for simulation."""

        clone = self.__class__.__new__(self.__class__)

        # Core configuration/state shared across clones.
        clone.num_players = self.num_players
        clone.starting_stack = self.starting_stack
        clone.player_strategies = self.player_strategies
        clone.history_limit = self.history_limit
        clone.logger = self.logger
        clone.verbose = self.verbose

        # Mutable per-hand state.
        clone.rules = self.rules.clone()
        clone.end_game_early = self.end_game_early
        clone.hand_count = self.hand_count
        clone.current_hand_initial_actions = list(self.current_hand_initial_actions)
        clone.historical_actions = self.historical_actions

        def _copy_winner(value):
            if isinstance(value, list):
                return list(value)
            return value

        clone.winner = _copy_winner(self.winner)
        clone.last_winner = _copy_winner(self.last_winner)

        return clone

    def initialize_game(self):
        # Reset game state for new hand
        self.last_winner = None
        self.rules.hands = [[] for _ in range(self.num_players)]
        self.rules.community_cards = []
        self.rules.pot = 0
        self.rules.bets = [0] * self.num_players
        self.rules.current_bet = 0
        self.rules.previous_raise_amount = 0
        self.rules.betting_history = []
        self.rules.actions_this_round = 0
        self.rules.last_raiser = None
        self.rules.total_bets_this_hand = [0] * self.num_players

        # Reset and shuffle deck
        self.rules.deck = self.rules._create_deck()
        self.rules.shuffle_deck()

        self.current_hand_initial_actions = []  # Reset at the start of each hand
        structured_blinds = self.rules.post_blinds()
        self.current_hand_initial_actions = structured_blinds
        self.deal_hands()
        self.hand_count += 1
        self.historical_actions.append(
            {
                "hand_number": self.hand_count,
                "dealer": self.rules.dealer_button + 1,
                "actions": self.rules.betting_history.copy(),
                "community_cards": self.format_hand_display(self.rules.community_cards),
                "pot": self.rules.pot,
                "players": self.get_player_status(),
            }
        )

    def deal_hands(self):
        self.rules.deal()

    def play_stage(self, stage):
        self.rules.deal_community_cards(stage)
        self.rules.betting_round = stage

    def process_action(self, player_index, action, raise_amount=None):
        """Process a single player's action with correct reopen/short all-in semantics."""
        # This method now also handles updating self.rules.last_raiser if a bet/raise occurs.
        # It also updates self.rules.previous_raise_amount correctly.
        action_completed = False
        try:
            original_current_bet = self.rules.current_bet
            is_aggressive_action = False

            if action == "call":
                amount_to_call = self.rules.current_bet - self.rules.bets[player_index]
                if amount_to_call <= 0:  # Cannot call if no bet or already called
                    # This might happen if player tries to call when it's a check, or they are already matching current_bet
                    # Consider this a check if current_bet is 0 and bets[player_index] is 0
                    if self.rules.current_bet == 0 and self.rules.bets[player_index] == 0:
                        self.rules.betting_history.append((str(player_index), ("check", 0)))
                        self._log(f"Player {player_index + 1} checks.")
                        action_completed = True
                    else:  # Or if they are trying to call but already match the bet (e.g. after a previous partial all-in)
                        self._log(
                            f"Player {player_index + 1} effectively checks (already matching current bet or no bet to call)."
                        )
                        action_completed = True
                    # No change in bet needed if amount_to_call <=0
                elif amount_to_call > self.rules.player_chips[player_index]:  # All-in call
                    amount_to_call = self.rules.player_chips[player_index]
                    self.rules.betting_history.append((str(player_index), ("call", amount_to_call)))
                    self._log(
                        f"Player {player_index + 1} calls all-in with {amount_to_call} chips."
                    )
                    new_bet = self.rules.bets[player_index] + amount_to_call
                    self.rules.bet(
                        player_index, new_bet
                    )  # bet method handles chip deduction and all-in state
                    action_completed = True
                else:  # Regular call
                    self.rules.betting_history.append((str(player_index), ("call", amount_to_call)))
                    self._log(f"Player {player_index + 1} calls {amount_to_call} chips.")
                    new_bet = self.rules.bets[player_index] + amount_to_call
                    self.rules.bet(player_index, new_bet)
                    action_completed = True

            elif action == "bet" or action == "raise":
                # 'bet' is used when current_bet is 0. 'raise' is used when current_bet > 0.
                is_raise_action = self.rules.current_bet > 0

                amount_to_call = self.rules.current_bet - self.rules.bets[player_index]
                if amount_to_call < 0:
                    amount_to_call = 0  # Should not happen if logic is correct

                # If raise_amount is None, it's an error from strategy or it's an opening bet.
                # For an opening bet (current_bet is 0), raise_amount is the bet size.
                # For a raise (current_bet > 0), raise_amount is the additional amount on top of current_bet.

                if raise_amount is None:  # Should be caught by strategy, but as a fallback:
                    if is_raise_action:  # Trying to raise but no amount
                        raise ValueError("Raise amount must be specified for a raise.")
                    else:  # Trying to bet but no amount
                        raise ValueError("Bet amount must be specified for a bet.")

                # This returns the *additional* amount for a raise, or BB for an opening bet.
                min_bet_or_raise_value = self.get_min_raise_amount(player_index)

                actual_raise_or_bet_amount = raise_amount

                if is_raise_action:  # This is a RAISE
                    if actual_raise_or_bet_amount < min_bet_or_raise_value:
                        self._log(
                            f"Player {player_index + 1} attempted to raise by {actual_raise_or_bet_amount}, less than min raise of {min_bet_or_raise_value}. Adjusting to min raise."
                        )
                        actual_raise_or_bet_amount = min_bet_or_raise_value

                    total_player_bet = self.rules.current_bet + actual_raise_or_bet_amount

                else:  # This is an opening BET
                    if (
                        actual_raise_or_bet_amount < min_bet_or_raise_value
                    ):  # min_bet_or_raise_value is BB here
                        self._log(
                            f"Player {player_index + 1} attempted to bet {actual_raise_or_bet_amount}, less than min bet of {min_bet_or_raise_value}. Adjusting to min bet."
                        )
                        actual_raise_or_bet_amount = min_bet_or_raise_value
                    total_player_bet = actual_raise_or_bet_amount

                # Check if player has enough chips for the full intended bet/raise
                goes_all_in = (
                    total_player_bet - self.rules.bets[player_index]
                ) > self.rules.player_chips[player_index]
                if goes_all_in:
                    # Player is going all-in (may be short of a full min raise)
                    all_in_amount = (
                        self.rules.player_chips[player_index] + self.rules.bets[player_index]
                    )
                    self._log(
                        f"Player {player_index + 1} goes all-in with {self.rules.player_chips[player_index]} chips (total bet {all_in_amount})."
                    )
                    total_player_bet = all_in_amount  # This is their all-in bet amount
                    # The actual_raise_or_bet_amount needs to be recalculated if they are all-in short
                    if total_player_bet > self.rules.current_bet:
                        actual_raise_or_bet_amount = total_player_bet - self.rules.current_bet
                    else:
                        # All-in is just a call or less
                        actual_raise_or_bet_amount = 0

                else:  # Sufficient chips for the bet/raise
                    if is_raise_action:
                        self._log(
                            f"Player {player_index + 1} raises by {actual_raise_or_bet_amount} to {total_player_bet} chips."
                        )
                    else:  # Opening bet
                        self._log(f"Player {player_index + 1} bets {total_player_bet} chips.")

                if is_raise_action:
                    self.rules.betting_history.append(
                        (str(player_index), ("raise", actual_raise_or_bet_amount))
                    )
                else:
                    self.rules.betting_history.append(
                        (str(player_index), ("bet", total_player_bet))
                    )

                # Call self.rules.bet with the player's total bet for this round
                self.rules.bet(player_index, total_player_bet)
                action_completed = True

                # Decide if this action *reopens* action:
                # Only if there was an actual raise of at least the minimum increment.
                reopened = False
                if self.rules.current_bet > original_current_bet:
                    # Compute increment and check minimum raise increment
                    inc = self.rules.current_bet - original_current_bet
                    min_inc = self.get_min_raise_amount(player_index)
                    # A short all-in (stack-constrained) that doesn't reach min_inc does NOT reopen action.
                    if inc >= min_inc and not goes_all_in:
                        reopened = True

                if reopened:
                    self.rules.last_raiser = player_index
                    # previous_raise_amount should be the amount the bet *increased by*
                    self.rules.previous_raise_amount = self.rules.current_bet - original_current_bet
                    is_aggressive_action = True
                elif (
                    total_player_bet == original_current_bet and original_current_bet > 0
                ):  # Matched the current bet, but was it an "aggressive" all-in?
                    # This case can happen if someone goes all-in for less than a full raise, but it's still a raise.
                    # The self.rules.bet method updates current_bet only if total_player_bet > self.current_bet.
                    # So, if total_player_bet IS the new current_bet, it means it was aggressive.
                    # The logic in self.rules.bet handles setting self.current_bet.
                    # We check if self.rules.current_bet changed.
                    # No, this is simpler: if their new bet > original_current_bet, it's aggressive.
                    # The self.rules.bet already updated self.current_bet and self.previous_raise_amount.
                    # We just need to set self.rules.last_raiser.
                    # This is already handled by the if self.rules.current_bet > original_current_bet check.
                    pass

            elif action == "fold":
                self.rules.active_players[player_index] = False
                self.rules.betting_history.append((str(player_index), ("fold", None)))
                self._log(f"Player {player_index + 1} folds.")
                active_players_count = sum(
                    1 for i in range(self.num_players) if self.rules.active_players[i]
                )
                if active_players_count == 1:
                    self.end_game_early = True
                    # Find the only remaining player
                    for i in range(self.num_players):
                        if self.rules.active_players[i]:
                            self.winner = i
                            break
                action_completed = True
            elif action == "check":
                # Check is only allowed if current_bet is 0 or player's bet matches current_bet
                if (
                    self.rules.current_bet == 0
                    or self.rules.bets[player_index] == self.rules.current_bet
                ):
                    self.rules.betting_history.append((str(player_index), ("check", 0)))
                    self._log(f"Player {player_index + 1} checks.")
                    action_completed = True
                else:  # Bet to call, cannot check
                    self._log(
                        f"Player {player_index + 1} tried to check, but there is a bet of {self.rules.current_bet - self.rules.bets[player_index]} to call. Defaulting to fold."
                    )
                    self.process_action(player_index, "fold")  # Or 'call' if preferred default
                    return
            else:  # Invalid action string
                self._log(
                    f"Player {player_index + 1} made an invalid action '{action}' and is folding by default."
                )
                self.process_action(player_index, "fold")
                return
        except ValueError as ve:
            self._log(
                f"Error processing action for Player {player_index + 1} ('{action}', {raise_amount}): {ve}. Defaulting to fold."
            )
            # Fallback action, typically fold.
            if self.rules.active_players[
                player_index
            ]:  # Ensure not trying to fold an already folded player
                self.process_action(player_index, "fold")
            return

        if action_completed:
            try:
                self.rules.actions_this_round += 1
            except AttributeError:  # pragma: no cover - defensive
                pass

    def betting_round(self, is_preflop=False):
        """
        Manages a single betting round (pre-flop, flop, turn, or river).
        Action continues until all players have acted and all bets are settled,
        or until only one player remains.
        """
        self.rules.actions_this_round = 0  # Reset for the current betting round.
        self.rules.last_raiser = None  # Reset last raiser for the round

        if not is_preflop:
            # Post-flop: action starts with the first active player left of the dealer.
            # Bets from previous street are consolidated into the pot; current round bets start at 0.
            self.rules.current_bet = 0
            self.rules.bets = [0] * self.num_players  # Bets for this specific street.
            self.rules.previous_raise_amount = 0  # Reset for the new street.

            # Determine starting player for post-flop rounds
            current_player_idx = (self.rules.dealer_button + 1) % self.num_players
            while not (
                self.rules.active_players[current_player_idx]
                and self.rules.player_chips[current_player_idx] > 0
            ):
                current_player_idx = (current_player_idx + 1) % self.num_players
                if (
                    current_player_idx == (self.rules.dealer_button + 1) % self.num_players
                ):  # Full circle
                    break  # Avoid infinite loop if all are all-in or out
            self.rules.current_player = current_player_idx
        else:
            # Pre-flop: current_player is already set by post_blinds() to UTG.
            # self.rules.current_bet is Big Blind. self.rules.bets has blinds posted.
            # The player who "opened" the betting or made the last raise that others must respond to.
            # Initially, this is the Big Blind player, as their blind is the current bet.
            self.rules.last_raiser = (self.rules.dealer_button + 2) % self.num_players

        acted_in_sequence = [False] * self.num_players
        actions_taken_this_sequence = 0

        # Determine how many players can actually make a decision (not folded, not all-in already matching current bet)
        def count_players_able_to_act():
            count = 0
            for i in range(self.num_players):
                if self.rules.active_players[i] and self.rules.player_chips[i] > 0:
                    count += 1
                # Also count players who are all-in but haven't yet "acted" on the current bet level
                # (e.g. if someone raised after they went all-in for less)
                elif (
                    self.rules.active_players[i]
                    and self.rules.player_chips[i] == 0
                    and self.rules.bets[i] < self.rules.current_bet
                ):
                    count += 1
            return count

        num_players_to_act_in_sequence = count_players_able_to_act()

        while True:
            num_active_players = sum(
                1 for i in range(self.num_players) if self.rules.active_players[i]
            )
            if (
                num_active_players <= 1 and actions_taken_this_sequence > 0
            ):  # Game might end if only one player left
                if self.end_game_early:
                    break  # Fold led to one player
                # If one active player, and all bets are settled (e.g. they made a bet everyone folded to)
                is_settled = True
                for i in range(self.num_players):
                    if (
                        self.rules.active_players[i]
                        and self.rules.player_chips[i] > 0
                        and self.rules.bets[i] != self.rules.current_bet
                    ):
                        is_settled = False
                        break
                if is_settled:
                    break

            player_index = self.rules.current_player

            # Check if round should end:
            # All active players (who are not all-in for less) have bet an equal amount,
            # AND (everyone has acted OR action is back to the last aggressor who doesn't re-open).
            if actions_taken_this_sequence >= num_players_to_act_in_sequence:
                bets_are_settled = True
                for i in range(self.num_players):
                    if (
                        self.rules.active_players[i] and self.rules.player_chips[i] > 0
                    ):  # If active and not all-in
                        if self.rules.bets[i] != self.rules.current_bet:
                            bets_are_settled = False
                            break
                if bets_are_settled:
                    # Special pre-flop BB case: if no raise yet and action is on BB who hasn't acted on the BB.
                    is_bb_preflop_check_option = (
                        is_preflop
                        and player_index == (self.rules.dealer_button + 2) % self.num_players
                        and self.rules.current_bet == self.rules.big_blind
                        and not acted_in_sequence[player_index]
                    )

                    if not is_bb_preflop_check_option:
                        break  # Betting round is over

            current_player_is_active = self.rules.active_players[player_index]
            current_player_has_chips = self.rules.player_chips[player_index] > 0
            # Player must act if active AND (has chips OR is all-in for less than current bet)
            needs_to_act = current_player_is_active and (
                current_player_has_chips
                or (
                    self.rules.bets[player_index] < self.rules.current_bet
                    and self.rules.player_chips[player_index] == 0
                )
            )

            if not needs_to_act or acted_in_sequence[player_index]:
                self.rules.advance_turn()
                # If we advanced past the player who was supposed to close the action, and they already acted, round might be over.
                # This is caught by the main loop condition (actions_taken_this_sequence >= num_players_to_act_in_sequence and bets_are_settled)
                continue

            # Player acts
            strategy = self.player_strategies[player_index]
            action, raise_amount = strategy.choose_action(self, player_index)

            original_current_bet_level = self.rules.current_bet

            self.process_action(player_index, action, raise_amount)
            acted_in_sequence[player_index] = True
            actions_taken_this_sequence += 1

            if self.end_game_early:  # e.g., everyone else folded during process_action
                break

            # If action was a bet or raise that changed the current_bet level
            if self.rules.current_bet > original_current_bet_level or (
                action in ["bet", "raise"] and self.rules.last_raiser == player_index
            ):  # last_raiser is set in process_action
                # Action has been re-opened. Reset sequence for other players.
                actions_taken_this_sequence = (
                    1  # The current player is the first to act in this new sequence.
                )
                acted_in_sequence = [False] * self.num_players
                acted_in_sequence[player_index] = True  # This player has acted.
                num_players_to_act_in_sequence = count_players_able_to_act()
                if self.rules.last_raiser == player_index:  # if they raised/bet
                    num_players_to_act_in_sequence = (
                        count_players_able_to_act()
                    )  # All others need to act
                # If player just called a raise, sequence does not reset.

            self.rules.advance_turn()

        # After the loop, the betting round for this street is over.
        # Consolidate bets into pot is done by self.rules.bet().
        # Prepare for the *next* street or showdown by cleaning up bets for *this* street.
        if not self.end_game_early:  # Don't cleanup if game ended, winner takes pot as is.
            self.rules.end_betting_round_cleanup()

    def get_min_raise_amount(self, player_index):
        """Return the minimum additional chips required to raise.

        The calculation mirrors official WSOP rules: a raise must be at least the
        size of the previous bet or raise.  We reuse ``min_raise_to`` from the
        reference rules module to ensure consistency.
        """
        min_total = min_raise_to(
            self.rules.current_bet,
            self.rules.previous_raise_amount,
            self.rules.big_blind,
        )
        return min_total - self.rules.current_bet

    def get_payoff(self, player_id):
        if not self.is_hand_over():
            return 0

        # Case 1: Everyone else folded
        active_players = [i for i, active in enumerate(self.rules.active_players) if active]
        if len(active_players) == 1:
            if active_players[0] == player_id:
                # This player won the pot
                return self.rules.pot - self.rules.total_bets_this_hand[player_id]
            else:
                # This player folded
                return -self.rules.total_bets_this_hand[player_id]

        # Case 2: Showdown
        winnings, _ = self.perform_showdown()
        return winnings.get(player_id, 0) - self.rules.total_bets_this_hand[player_id]

    def get_max_raise_amount(self, player_index):
        # This is the total amount a player can raise TO, not the additional amount.
        # Max raise is effectively all their chips.
        # The 'raise_amount' in process_action is the additional amount.
        # So max additional raise is player_chips.
        return self.rules.player_chips[player_index]

    # ---------------------------
    # Showdown / Side-pot helpers
    # ---------------------------
    def _hand_strength(
        self, hole: list[str], board: list[str]
    ) -> tuple[int, list[str], int]:
        """Return a comparable hand strength and rank class.

        Lower scores are better for ``treys``' ``Evaluator`` implementation.  The
        returned tuple is ``(score, tiebreaker_cards_sorted, hand_rank_class)`` so
        ties can be handled deterministically while also exposing the winning hand
        class for logging purposes."""
        try:
            evaluator = Evaluator()
            hole_cards = [Card.new(c) for c in hole]
            board_cards = [Card.new(c) for c in board]
            score = evaluator.evaluate(board_cards, hole_cards)
            rank_class = evaluator.get_rank_class(score)
        except Exception:
            rank_order = {
                r: i
                for i, r in enumerate(
                    ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
                )
            }
            all7 = hole + board
            all7_sorted = sorted(all7, key=lambda x: (rank_order[x[0]], x[1]), reverse=True)
            score = -sum((rank_order[c[0]] + 2) for c in all7_sorted[:5])
            rank_class = 0
        return score, sorted(hole + board), rank_class

    def _compute_side_pots(self) -> list[dict[str, object]]:
        """Compute side pots using the shared helper from :mod:`gatc_poker`."""

        contributions: dict[int, int] = {}
        for idx, amount in enumerate(self.rules.total_bets_this_hand):
            chips = int(amount)
            if chips > 0:
                contributions[idx] = chips
        in_showdown = {
            idx for idx, active in enumerate(self.rules.active_players) if active
        }
        pots = gatc_compute_side_pots(contributions, in_showdown)
        return [
            {"amount": pot.amount, "eligible": set(pot.eligible)}
            for pot in pots
        ]

    def perform_showdown(self) -> tuple[dict[int, int], dict[int, int]]:
        """Evaluate all active players' hands and build side pots.

        Returns a tuple ``(winnings, best_hands)`` where ``winnings`` maps each
        player index to the number of chips they receive from all pots and
        ``best_hands`` maps each active player to the evaluator's rank class for
        their best five-card hand."""
        board = self.rules.community_cards
        active = [i for i, a in enumerate(self.rules.active_players) if a]
        scores: dict[int, tuple[int, list[str]]] = {}
        best_hands: dict[int, int] = {}
        for i in active:
            hole = self.rules.hands[i]
            score, tiebreaker, rank_class = self._hand_strength(hole, board)
            scores[i] = (score, tiebreaker)
            best_hands[i] = rank_class

        pots = self._compute_side_pots()
        if not pots:
            return {}, best_hands

        winnings: dict[int, int] = {i: 0 for i in range(self.num_players)}
        for pot in pots:
            amount = int(pot["amount"])
            elig = set(pot["eligible"])
            if not elig or amount <= 0:
                continue
            best_score = min(scores[i][0] for i in elig)
            winners = [i for i in elig if scores[i][0] == best_score]
            share, rem = divmod(amount, len(winners))
            for w in winners:
                winnings[w] += share
            start = (self.rules.dealer_button + 1) % self.num_players
            order = [((start + k) % self.num_players) for k in range(self.num_players)]
            for seat in order:
                if rem == 0:
                    break
                if seat in winners:
                    winnings[seat] += 1
                    rem -= 1
        return winnings, best_hands

    def is_hand_over(self) -> bool:
        """The hand is over if:
        (a) we ended early due to folds (single active player); or
        (b) five board cards are dealt AND the betting round is settled."""
        active_cnt = sum(1 for a in self.rules.active_players if a)
        if active_cnt <= 1 or self.end_game_early:
            return True
        if len(self.rules.community_cards) == 5:
            active_and_not_allin = [
                i
                for i in range(self.num_players)
                if self.rules.active_players[i] and self.rules.player_chips[i] > 0
            ]
            if all(self.rules.bets[i] == self.rules.current_bet for i in active_and_not_allin):
                return True
        return False

    def get_valid_actions(self, player_index):
        """Return a list of valid actions for the given player."""
        amount_to_call = self.rules.current_bet - self.rules.bets[player_index]

        # Player has no chips left
        if self.rules.player_chips[player_index] <= 0:
            if amount_to_call > 0:
                return ["call", "fold"]
            return ["check"]

        if amount_to_call > 0:
            if self.rules.player_chips[player_index] <= amount_to_call:
                return ["call", "fold"]
            else:
                return ["call", "raise", "fold"]
        else:
            if self.rules.player_chips[player_index] > 0:
                return ["check", "bet"]
            return ["check"]

    def reset_for_next_hand(self):
        """Prepare the game engine for the next hand.

        Chip stacks are **not** reset so players keep their winnings and losses
        like in a real game. Players with zero chips are marked as eliminated.
        """

        # Update active players based on remaining chips and rotate the dealer
        self.rules.active_players = [chips > 0 for chips in self.rules.player_chips]
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
        self.rules.total_bets_this_hand = [0] * self.num_players  # Reset for next hand
        self.end_game_early = False
        self.winner = None
        self._pending_showdown_winnings = None

    def play_game(self):
        self._log(f"--- Hand {self.hand_count + 1} ---")
        self.initialize_game()  # Deals hands, posts blinds, sets up pre-flop player
        self.show_player_hands()
        self.display_player_chips()

        # Pre-flop betting round
        self._log("\n--- Pre-flop Betting Round ---")
        self.betting_round(is_preflop=True)
        if self.end_game_early:
            self.declare_winner()  # Pot distribution happens here
            self.save_history_if_needed()
            self.reset_for_next_hand()  # Reset even if game ends early
            return

        # Flop, Turn, River stages
        stages = ["flop", "turn", "river"]
        for stage in stages:
            if sum(1 for i in range(self.num_players) if self.rules.active_players[i]) <= 1:
                # No betting round if only one player (or fewer) is left before dealing community cards for this stage
                break
            self.play_stage(stage)  # Deals community cards for the stage
            self.print_community_cards(stage)
            self.display_player_chips()
            self._log(f"\n--- Betting Round after the {stage.capitalize()} ---")
            self.betting_round(is_preflop=False)
            if self.end_game_early:
                self.declare_winner()
                self.save_history_if_needed()
                self.reset_for_next_hand()  # Reset even if game ends early
                return

        # Showdown
        self._log("\n--- Showdown ---")
        # Pot was already awarded if end_game_early was true.
        # If not end_game_early, then proceed to showdown.
        for i in range(self.num_players):
            if self.rules.active_players[i]:
                hand_str = self.format_hand_display(self.rules.hands[i])
                self._log(f"Player {i + 1}'s hand: {hand_str}")

        showdown_winnings, best_hands = self.perform_showdown()
        positive_winnings = {i: amt for i, amt in showdown_winnings.items() if amt > 0}
        if not positive_winnings:
            self._log("\nNo winner could be determined.")
        else:
            max_winnings = max(positive_winnings.values())
            winners = [i for i, amt in positive_winnings.items() if amt == max_winnings]
            self._pending_showdown_winnings = showdown_winnings
            self.winner = winners[0] if len(winners) == 1 else None
            self.show_winner(winners, best_hands)

        self.save_history_if_needed()

    def declare_winner(self):
        self._log("\n--- Hand Conclusion ---")
        self._log("All other players have folded.")
        self._log(f"The winner is Player {self.winner + 1}!")

        winner_player_index = self.winner
        pots = self._compute_side_pots()
        if not pots:
            actual_winnings = self.rules.pot
            self._log(f"Pot won: {actual_winnings} chips.")
            self.rules.player_chips[winner_player_index] += actual_winnings
            if self.historical_actions:
                self.historical_actions[-1]["winner"] = (
                    f"Player {winner_player_index + 1}"
                )
                self.historical_actions[-1]["pot_won"] = actual_winnings
            self.last_winner = winner_player_index
            self.rules.pot = max(0, self.rules.pot - actual_winnings)
            return

        start = (self.rules.dealer_button + 1) % self.num_players
        seat_order = [((start + k) % self.num_players) for k in range(self.num_players)]

        winnings_for_winner = 0
        redistributed: dict[int, int] = {i: 0 for i in range(self.num_players)}

        for idx, pot in enumerate(pots):
            amount = int(pot["amount"])
            if amount <= 0:
                continue

            eligible = set(pot["eligible"])
            pot_name = "main pot" if idx == 0 else f"side pot {idx}"

            if winner_player_index in eligible or not eligible:
                winnings_for_winner += amount
                if not eligible:
                    self._log(
                        f"{pot_name.capitalize()} ({amount} chips) defaults to Player "
                        f"{winner_player_index + 1} (no eligible opponents)."
                    )
                else:
                    self._log(
                        f"{pot_name.capitalize()} ({amount} chips) awarded to Player "
                        f"{winner_player_index + 1}."
                    )
                continue

            share, remainder = divmod(amount, len(eligible))
            pot_distribution: dict[int, int] = {pid: share for pid in eligible}
            if remainder:
                for seat in seat_order:
                    if remainder == 0:
                        break
                    if seat in eligible:
                        pot_distribution[seat] += 1
                        remainder -= 1

            for pid, amt in pot_distribution.items():
                redistributed[pid] += amt

            distribution_str = ", ".join(
                f"Player {pid + 1} receives {amt}"
                for pid, amt in sorted(pot_distribution.items())
            )
            self._log(
                f"{pot_name.capitalize()} ({amount} chips) redistributed among "
                f"eligible opponents: {distribution_str}."
            )

        if winnings_for_winner:
            self.rules.player_chips[winner_player_index] += winnings_for_winner
        self._log(f"Pot won: {winnings_for_winner} chips.")

        for pid, amt in sorted(redistributed.items()):
            if amt <= 0 or pid == winner_player_index:
                continue
            self.rules.player_chips[pid] += amt
            self._log(
                f"Player {pid + 1} receives {amt} chips from unresolved side pots."
            )

        total_distributed = winnings_for_winner + sum(
            amt for pid, amt in redistributed.items() if pid != winner_player_index
        )
        self.rules.pot = max(0, self.rules.pot - total_distributed)

        if self.historical_actions:
            self.historical_actions[-1]["winner"] = (
                f"Player {winner_player_index + 1}"
            )
            self.historical_actions[-1]["pot_won"] = winnings_for_winner
        self.last_winner = winner_player_index

    def show_winner(self, winner, player_best_hands):
        """Distribute showdown winnings and log the results."""

        evaluator = Evaluator()
        hand_rankings: dict[int, str] = {}
        for player_index, rank_class in player_best_hands.items():
            try:
                hand_rankings[player_index] = evaluator.class_to_string(rank_class)
            except Exception:
                hand_rankings[player_index] = "Unknown"

        payouts = self._pending_showdown_winnings or {}
        positive_payouts = {pid: amt for pid, amt in payouts.items() if amt > 0}

        winners_list = winner if isinstance(winner, list) else [winner]
        winners_list = [w for w in winners_list if positive_payouts.get(w, 0) > 0]

        if len(winners_list) > 1:
            self._log("\nIt's a tie between the following players:")
            for w in winners_list:
                hand_name = hand_rankings.get(w, "Unknown")
                self._log(f"Player {w + 1} with a {hand_name}")
            self._log("Pot split between players.")
        elif len(winners_list) == 1:
            w = winners_list[0]
            self._log(
                f"\nThe winner is Player {w + 1} with a {hand_rankings.get(w, 'Unknown')}!"
            )
        elif positive_payouts:
            self._log("\nMultiple side pots were awarded.")
        else:
            self._log("\nNo winner could be determined.")

        if not positive_payouts:
            self._pending_showdown_winnings = None
            self.reset_for_next_hand()
            return

        total_awarded = 0
        for pid in sorted(positive_payouts):
            amount = positive_payouts[pid]
            hand_name = hand_rankings.get(pid, "Unknown")
            self._log(f"Player {pid + 1} wins {amount} chips with a {hand_name}.")
            self.rules.player_chips[pid] += amount
            total_awarded += amount

        if self.historical_actions:
            winners_formatted = [f"Player {pid + 1}" for pid in sorted(positive_payouts)]
            if len(winners_formatted) == 1:
                self.historical_actions[-1]["winner"] = winners_formatted[0]
            else:
                self.historical_actions[-1]["winner"] = winners_formatted
            self.historical_actions[-1]["pot_won"] = total_awarded

        if len(positive_payouts) == 1:
            self.last_winner = next(iter(positive_payouts))
        else:
            self.last_winner = sorted(positive_payouts)

        self.rules.pot = max(0, self.rules.pot - total_awarded)
        self._pending_showdown_winnings = None

        self.reset_for_next_hand()

    def format_hand_display(self, hand):
        # Mapping from treys-compatible suits to display-friendly symbols
        suit_symbols = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
        return " ".join([card[0] + suit_symbols.get(card[1], card[1]) for card in hand])

    def display_player_chips(self):
        self._log("\n--- Player Chip Counts ---")
        for i in range(self.num_players):
            status = "Active" if self.rules.active_players[i] else "Eliminated"
            self._log(f"Player {i + 1}: {self.rules.player_chips[i]} chips ({status})")
        self._log("---------------------------\n")

    def show_player_hands(self):
        for i in range(self.num_players):
            strategy = self.player_strategies[i]
            if strategy.is_human:
                hand_str = self.format_hand_display(self.rules.hands[i])
                self._log(f"Player {i + 1}'s hand: {hand_str}")
            else:
                pass  # AI hands can be hidden or shown as desired

    def get_initial_state(self):
        """Return a simple tensor representation of the current game state."""
        import numpy as np

        state = np.zeros((self.num_players + 5, len(RANKS), len(SUITS)), dtype=int)
        for i, hand in enumerate(self.rules.hands):
            for card in hand:
                rank = RANKS.index(card[0])
                suit = SUITS.index(card[1])
                state[i, rank, suit] = 1
        for j, card in enumerate(self.rules.community_cards):
            rank = RANKS.index(card[0])
            suit = SUITS.index(card[1])
            state[self.num_players + j, rank, suit] = 1
        return np.expand_dims(state, axis=0)

    def print_community_cards(self, stage):
        community_str = self.format_hand_display(self.rules.community_cards)
        self._log(f"\nCommunity cards after the {stage.capitalize()}: {community_str}")
        self._log(f"Pot: {self.rules.pot} chips.")

    def get_player(self, player_id: str) -> MockPlayer:
        player_idx = int(player_id)
        player = MockPlayer(
            player_id=player_id,
            hand=self.rules.hands[player_idx],
            stack=self.rules.player_chips[player_idx],
        )
        player.current_bet_in_round = self.rules.bets[player_idx]
        return player

    def get_player_status(self) -> dict[str, dict[str, int | str]]:
        return {
            f"Player {i + 1}": {
                "chips": self.rules.player_chips[i],
                "status": "Active" if self.rules.active_players[i] else "Eliminated",
            }
            for i in range(self.num_players)
        }

    def save_history_if_needed(self) -> None:
        if self.hand_count % self.history_limit == 0:
            timestamp = datetime.now().strftime("%y%m%d%H%M%S")
            filename = f"data/historical_actions_{timestamp}.json"
            try:
                with open(filename, "w") as f:
                    json.dump(self.historical_actions, f, indent=4)
                self._log(f"\n--- Historical data saved to {filename} ---")
                self.historical_actions.clear()
            except Exception as e:
                self._log(f"Error saving historical data: {e}")
