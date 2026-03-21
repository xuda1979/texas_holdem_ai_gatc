from __future__ import annotations

"""Standalone Texas Hold'em rule management.

This module contains the :class:`TexasHoldemRules` implementation that was
previously embedded inside :mod:`poker_ai.engine.texas_holdem`.  Extracting the
rules into their own module keeps the engine orchestration (player strategies,
cash table integration, logging) decoupled from the low level betting and deck
mechanics.  The separation makes it easier to unit test the rules in isolation
and avoids importing heavy engine dependencies when they are not required.
"""

import logging

from poker_ai.engine.components import (
    BettingRoundTracker,
    CardDeck,
    DeckManager,
    PlayerManager,
    RANKS,
    SUITS,
)


class TexasHoldemRules:
    """Core game state and rule management for Texas Hold'em."""

    def __init__(self, num_players: int = 2, starting_stack: int = 10_000, verbose: bool = True):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.deck_manager = DeckManager()
        self.hands = [[] for _ in range(num_players)]
        self.community_cards: list[str] = []
        self.pot = 0
        self._bets = [0] * num_players
        self._player_chips = [starting_stack] * num_players
        self._active_players = [True] * num_players
        self._total_bets_this_hand = [0] * num_players
        self.player_manager = PlayerManager(
            player_chips=self._player_chips,
            bets=self._bets,
            total_bets_this_hand=self._total_bets_this_hand,
            active_players=self._active_players,
        )
        self._betting = BettingRoundTracker()
        self.small_blind = 10
        self.big_blind = 20
        self.current_bet = 0
        self.previous_raise_amount = 0
        self.dealer_button = 0
        self.betting_history: list[tuple[str, tuple[str, int | None]]] = []
        self.actions_this_round = 0
        self.last_raiser: int | None = None
        self.betting_round = "pre-flop"
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            self.logger.info(msg)

    # ------------------------------------------------------------------
    # Properties bridging legacy attributes and the new component helpers
    # ------------------------------------------------------------------
    @property
    def deck(self):
        return self.deck_manager.cards

    @deck.setter
    def deck(self, cards):
        self.deck_manager.cards = CardDeck(cards)

    @property
    def bets(self):
        return self._bets

    @bets.setter
    def bets(self, values):
        self._bets = list(values)
        if hasattr(self, "player_manager"):
            self.player_manager.update_references(bets=self._bets)

    @property
    def player_chips(self):
        return self._player_chips

    @player_chips.setter
    def player_chips(self, values):
        self._player_chips = list(values)
        if hasattr(self, "player_manager"):
            self.player_manager.update_references(player_chips=self._player_chips)

    @property
    def active_players(self):
        return self._active_players

    @active_players.setter
    def active_players(self, values):
        self._active_players = list(values)
        if hasattr(self, "player_manager"):
            self.player_manager.update_references(active_players=self._active_players)

    @property
    def total_bets_this_hand(self):
        return self._total_bets_this_hand

    @total_bets_this_hand.setter
    def total_bets_this_hand(self, values):
        self._total_bets_this_hand = list(values)
        if hasattr(self, "player_manager"):
            self.player_manager.update_references(
                total_bets_this_hand=self._total_bets_this_hand
            )

    @property
    def current_bet(self):
        return self._betting.current_bet

    @current_bet.setter
    def current_bet(self, value):
        self._betting.current_bet = value

    @property
    def previous_raise_amount(self):
        return self._betting.previous_raise_amount

    @previous_raise_amount.setter
    def previous_raise_amount(self, value):
        self._betting.previous_raise_amount = value

    @property
    def actions_this_round(self):
        return self._betting.actions_this_round

    @actions_this_round.setter
    def actions_this_round(self, value):
        self._betting.actions_this_round = value

    @property
    def last_raiser(self):
        return self._betting.last_raiser

    @last_raiser.setter
    def last_raiser(self, value):
        self._betting.last_raiser = value

    # ------------------------------------------------------------------
    # High level lifecycle helpers
    # ------------------------------------------------------------------
    def reset_for_new_hand(self) -> None:
        """Reset mutable state prior to dealing a fresh hand."""

        self.hands = [[] for _ in range(self.num_players)]
        self.community_cards = []
        self.pot = 0
        self.betting_history = []
        self.player_manager.reset_for_new_hand()
        self.current_bet = 0
        self.previous_raise_amount = 0
        self.actions_this_round = 0
        self.last_raiser = None
        self.deck_manager.reset()

    def _create_deck(self):
        return CardDeck(self.deck_manager.build_fresh_deck())

    def _next_player_with_chips(self, start_index: int, *, include_start: bool = False) -> int | None:
        """Return the next active seat with chips starting after ``start_index``."""

        return self.player_manager.next_player_with_chips(start_index, include_start=include_start)

    def shuffle_deck(self) -> None:
        self.deck_manager.shuffle()

    def rotate_dealer(self) -> None:
        """Move the dealer button to the next player."""

        self.dealer_button = (self.dealer_button + 1) % self.num_players

    def deal(self) -> None:
        self.deck_manager.deal_hole_cards(self.hands, self.active_players)

    def post_blinds(self) -> list[tuple[str, tuple[str, int]]]:
        structured_blind_actions: list[tuple[str, tuple[str, int]]] = []
        small_blind_player = self._next_player_with_chips(self.dealer_button)
        if small_blind_player is None:
            return []

        big_blind_player = self._next_player_with_chips(small_blind_player)
        if big_blind_player is None:
            big_blind_player = small_blind_player

        def _post_blind(player_index: int, blind_amount: int, label: str) -> int:
            contribution = self.player_manager.post_blind(player_index, blind_amount)
            self.pot += contribution
            self.betting_history.append((str(player_index), ("bet", contribution)))
            if contribution < blind_amount:
                self._log(
                    f"Player {player_index + 1} posts {label} blind of {contribution} chips (all-in)."
                )
            else:
                self._log(
                    f"Player {player_index + 1} posts {label} blind of {blind_amount} chips."
                )
            structured_blind_actions.append((str(player_index), ("bet", contribution)))
            return contribution

        _post_blind(small_blind_player, self.small_blind, "small")

        big_blind_contribution = _post_blind(big_blind_player, self.big_blind, "big")

        self.current_bet = big_blind_contribution
        self.previous_raise_amount = big_blind_contribution

        next_player = self._next_player_with_chips(big_blind_player)
        if next_player is None:
            next_player = big_blind_player
        self.current_player = next_player
        return structured_blind_actions

    def bet(self, player_index: int, amount: int) -> None:
        bet_difference = amount - self.bets[player_index]
        if amount < self.current_bet:
            raise ValueError("Bet amount is less than the current bet.")
        contribution, fully_applied = self.player_manager.apply_bet(player_index, amount)
        if contribution:
            self.pot += contribution

        actual_total = self.bets[player_index]
        previous_total = actual_total - contribution

        if not fully_applied and bet_difference > contribution:
            self._log(f"Player {player_index + 1} goes all-in with {contribution} chips.")
        if actual_total > self.current_bet and actual_total > previous_total:
            self._log(f"Player {player_index + 1} raises to {actual_total} chips.")
            previous_bet = self.current_bet
            self.previous_raise_amount = actual_total - previous_bet
            self.current_bet = actual_total
            self._log(f"Current bet is now {self.current_bet} chips.")

    def advance_turn(self) -> None:
        self.current_player = (self.current_player + 1) % self.num_players
        for _ in range(self.num_players):
            if self.active_players[self.current_player] and self.player_chips[self.current_player] > 0:
                break
            self.current_player = (self.current_player + 1) % self.num_players

    def reset_bets(self) -> None:
        self.player_manager.reset_bets_for_round()
        self.current_bet = 0
        self.previous_raise_amount = 0
        self.actions_this_round = 0

    def betting_round_is_over(self) -> bool:
        """Return True if all active players have matched the current bet."""

        active_players = self.player_manager.active_with_chips()
        if not active_players:
            return True
        all_settled = all(self.bets[i] == self.current_bet for i in active_players)
        return all_settled and self.actions_this_round >= len(active_players)

    def end_betting_round_cleanup(self) -> None:
        """Reset betting state for the start of a new street."""

        self.reset_bets()

    def deal_community_cards(self, round_stage: str) -> None:
        if round_stage == "flop":
            burned = self.deck_manager.burn()
            self._log(f"Burned a card: {self._format_card(burned)}.")
            for card in self.deck_manager.draw(3):
                self.community_cards.append(card)
                self._log(f"Dealt community card: {self._format_card(card)}.")
        elif round_stage == "turn":
            burned = self.deck_manager.burn()
            self._log(f"Burned a card: {self._format_card(burned)}.")
            card = self.deck_manager.draw()[0]
            self.community_cards.append(card)
            self._log(f"Dealt community card: {self._format_card(card)}.")
        elif round_stage == "river":
            burned = self.deck_manager.burn()
            self._log(f"Burned a card: {self._format_card(burned)}.")
            card = self.deck_manager.draw()[0]
            self.community_cards.append(card)
            self._log(f"Dealt community card: {self._format_card(card)}.")
        else:
            raise ValueError("Invalid round stage.")

    def _format_card(self, card: str) -> str:
        rank = card[0]
        suit = card[1]
        suit_symbols = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
        return f"{rank}{suit_symbols.get(suit, suit)}"

    def clone(self) -> "TexasHoldemRules":
        """Return a lightweight copy of the mutable rule state."""

        clone = self.__class__.__new__(self.__class__)
        clone.deck_manager = self.deck_manager.clone()
        clone._betting = self._betting.clone()

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

        clone.deck = CardDeck(self.deck)
        clone.hands = [list(hand) for hand in self.hands]
        clone.community_cards = list(self.community_cards)
        clone.bets = list(self.bets)
        clone.player_chips = list(self.player_chips)
        clone.active_players = list(self.active_players)
        clone.betting_history = list(self.betting_history)
        clone.total_bets_this_hand = list(self.total_bets_this_hand)
        clone.player_manager = self.player_manager.clone(
            player_chips=clone._player_chips,
            bets=clone._bets,
            total_bets_this_hand=clone._total_bets_this_hand,
            active_players=clone._active_players,
        )

        if hasattr(self, "current_player"):
            clone.current_player = self.current_player

        return clone


__all__ = ["TexasHoldemRules", "RANKS", "SUITS"]

