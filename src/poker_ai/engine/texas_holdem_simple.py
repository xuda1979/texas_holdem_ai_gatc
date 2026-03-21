from __future__ import annotations

import random
from collections import Counter

import numpy as np

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
DECK = [rank + suit for suit in SUITS for rank in RANKS]


class TexasHoldem:
    def __init__(self, num_players):
        self.num_players = num_players
        self.deck = DECK.copy()
        self.community_cards = []
        self.players_hands: list[list[str]] = []
        self.players_active = [True] * num_players
        self.bets = [0] * num_players
        self.current_bet = 0
        self.pot = 0
        self.betting_round = 0
        self.reset()

    def reset(self):
        self.deck = DECK.copy()
        random.shuffle(self.deck)
        self.community_cards = []
        self.players_hands = [[] for _ in range(self.num_players)]
        for _ in range(2):
            for player in range(self.num_players):
                self.players_hands[player].append(self._draw_card())
        self.players_active = [True] * self.num_players
        self.bets = [0] * self.num_players
        self.current_bet = 0
        self.pot = 0
        self.betting_round = 0
        self.display_stage("Pre-Flop")

    def _draw_card(self) -> str:
        if not self.deck:
            raise ValueError("The deck is empty; cannot draw more cards.")
        return self.deck.pop()

    def _burn_card(self) -> None:
        if not self.deck:
            raise ValueError("The deck is empty; cannot burn a card.")
        self.deck.pop()

    def get_initial_state(self):
        state = np.zeros((self.num_players + 5, len(RANKS), len(SUITS)))  # Shape (7, 13, 4)

        for i, hand in enumerate(self.players_hands):
            for card in hand:
                rank = RANKS.index(card[0])
                suit = SUITS.index(card[1])
                state[i, rank, suit] = 1

        for j, card in enumerate(self.community_cards):
            rank = RANKS.index(card[0])
            suit = SUITS.index(card[1])
            state[self.num_players + j, rank, suit] = 1

        # Add batch dimension (1, 7, 13, 4)
        state = np.expand_dims(state, axis=0)

        return state

    def display_stage(self, stage_name):
        print(f"\n--- {stage_name} ---")
        if self.community_cards:
            print(f"Community Cards: {self.community_cards}")
        for i, hand in enumerate(self.players_hands):
            if self.players_active[i]:
                print(f"Player {i+1} Hand: {hand}")
            else:
                print(f"Player {i+1} is folded.")
        print(f"Current Bet: {self.current_bet}")
        print(f"Pot: {self.pot}")

    def deal_flop(self):
        self._burn_card()
        self.community_cards.extend(self._draw_card() for _ in range(3))
        self.betting_round += 1
        self.display_stage("Flop")

    def deal_turn(self):
        self._burn_card()
        self.community_cards.append(self._draw_card())
        self.betting_round += 1
        self.display_stage("Turn")

    def deal_river(self):
        self._burn_card()
        self.community_cards.append(self._draw_card())
        self.betting_round += 1
        self.display_stage("River")

    def next_betting_round(self):
        """Move to the next betting round or stage of the game."""
        self.current_bet = 0  # Reset current bet for the new round
        self.bets = [0] * self.num_players
        if self.betting_round == 0:
            self.deal_flop()
        elif self.betting_round == 1:
            self.deal_turn()
        elif self.betting_round == 2:
            self.deal_river()
        elif self.betting_round == 3:
            self.betting_round += 1  # All betting rounds completed

    def apply_action(self, player_id, action, amount=0):
        if action == "fold":
            self.players_active[player_id] = False
            print(f"Player {player_id+1} folds.")
        elif action == "call":
            self.pot += self.current_bet - self.bets[player_id]
            self.bets[player_id] = self.current_bet
            print(f"Player {player_id+1} calls.")
        elif action == "raise":
            self.current_bet += amount
            self.pot += self.current_bet - self.bets[player_id]
            self.bets[player_id] = self.current_bet
            print(f"Player {player_id+1} raises to {self.current_bet}.")
        elif action == "check":
            print(f"Player {player_id+1} checks.")

    def is_terminal(self):
        """Determine if the game is in a terminal state."""
        active_players = sum(self.players_active)
        return active_players <= 1 or self.betting_round > 3

    def clone(self):
        """Create a deep copy of the game state."""
        cloned_game = TexasHoldem(self.num_players)
        cloned_game.deck = self.deck[:]
        cloned_game.community_cards = self.community_cards[:]
        cloned_game.players_hands = [hand[:] for hand in self.players_hands]
        cloned_game.players_active = self.players_active[:]
        cloned_game.bets = self.bets[:]
        cloned_game.current_bet = self.current_bet
        cloned_game.pot = self.pot
        cloned_game.betting_round = self.betting_round
        return cloned_game

    def evaluate_hand(self, hand):
        """Evaluate the strength of a given hand."""
        all_cards = hand + self.community_cards
        hand_combinations = self.get_hand_combinations(all_cards)
        best_hand = max(hand_combinations, key=self.hand_rank)
        return self.hand_rank(best_hand)

    def get_hand_combinations(self, cards):
        """Generate all possible 5-card combinations from the given cards."""
        if len(cards) < 5:
            return []
        combinations = []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                for k in range(j + 1, len(cards)):
                    for l in range(k + 1, len(cards)):
                        for m in range(l + 1, len(cards)):
                            combinations.append([cards[i], cards[j], cards[k], cards[l], cards[m]])
        return combinations

    def hand_rank(self, hand):
        """Determine the rank of a hand.

        The original implementation assumed each ``card`` in ``hand`` was a
        tuple of ``(rank, suit)`` values. In this project the simplified engine
        represents cards as strings such as ``"T♠"`` or ``"A♥"`` which caused a
        ``ValueError`` when unpacking the string into ``(rank, suit)``. The
        broken unpacking also meant all ranking logic silently failed. This
        method now accepts either tuple or string representations and performs
        a full hand evaluation using the conventional poker ranking rules.
        """

        rank_lookup = {rank: idx for idx, rank in enumerate("23456789TJQKA")}

        rank_values: list[int] = []
        suits: list[str] = []

        for card in hand:
            if isinstance(card, str):
                if len(card) < 2:
                    raise ValueError(f"Invalid card representation: {card!r}")
                rank_symbol, suit_symbol = card[0], card[-1]
            else:
                try:
                    rank_symbol, suit_symbol = card  # type: ignore[misc]
                except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Invalid card representation: {card!r}"
                    ) from exc

            if rank_symbol not in rank_lookup:
                raise ValueError(f"Unknown rank symbol: {rank_symbol!r}")

            rank_values.append(rank_lookup[rank_symbol])
            suits.append(str(suit_symbol))

        rank_values_sorted = sorted(rank_values, reverse=True)
        rank_counter = Counter(rank_values)
        counts_sorted = sorted(
            rank_counter.items(), key=lambda item: (-item[1], -item[0])
        )
        counts = tuple(count for _, count in counts_sorted)
        values = tuple(rank for rank, _ in counts_sorted)

        unique_ranks = sorted(rank_counter)
        straight_high: int | None = None
        if len(unique_ranks) == 5:
            if unique_ranks == [0, 1, 2, 3, 12]:
                straight_high = 3  # Five-high straight (wheel)
            elif unique_ranks[-1] - unique_ranks[0] == 4:
                straight_high = unique_ranks[-1]

        is_straight = straight_high is not None
        is_flush = len(set(suits)) == 1

        if is_straight and is_flush:
            if straight_high == rank_lookup["A"]:
                return (10,)
            return (9, straight_high)
        if counts == (4, 1):
            return (8, values[0], values[1])
        if counts == (3, 2):
            return (7, values[0], values[1])
        if is_flush:
            return (6, tuple(rank_values_sorted))
        if is_straight:
            return (5, straight_high)
        if counts == (3, 1, 1):
            return (4, values)
        if counts == (2, 2, 1):
            return (3, values)
        if counts == (2, 1, 1, 1):
            return (2, values)
        return (1, tuple(rank_values_sorted))

    def get_winner(self):
        """Determine the winner of the game."""
        best_rank = (-1,)
        winners = []
        for i, hand in enumerate(self.players_hands):
            if self.players_active[i]:
                rank = self.evaluate_hand(hand)
                if rank > best_rank:
                    best_rank = rank
                    winners = [i]
                elif rank == best_rank:
                    winners.append(i)
        return winners
