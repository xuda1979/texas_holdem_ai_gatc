from __future__ import annotations

import pytest

from poker_ai.engine.components import (
    BettingRoundTracker,
    DeckManager,
    PlayerManager,
    RANKS,
    SUITS,
)


class TestDeckManager:
    def test_reset_produces_full_deck(self):
        manager = DeckManager()
        manager.shuffle()
        manager.reset()
        assert len(manager.cards) == len(SUITS) * len(RANKS)
        assert sorted(manager.cards) == sorted(manager.build_fresh_deck())

    def test_deal_hole_cards_respects_active_flags(self):
        manager = DeckManager()
        hands = [[] for _ in range(3)]
        manager.deal_hole_cards(hands, [True, False, True])
        assert all(len(hand) == 2 for idx, hand in enumerate(hands) if idx != 1)
        assert hands[1] == []
        assert len(manager.cards) == len(manager.build_fresh_deck()) - 4

    def test_draw_and_burn_errors_when_empty(self):
        manager = DeckManager()
        manager.cards = []
        with pytest.raises(ValueError):
            manager.draw()
        with pytest.raises(ValueError):
            manager.burn()


class TestPlayerManager:
    def setup_method(self):
        self.player_chips = [100, 50, 0]
        self.bets = [0, 0, 0]
        self.totals = [0, 0, 0]
        self.active = [True, True, True]
        self.manager = PlayerManager(self.player_chips, self.bets, self.totals, self.active)

    def test_reset_for_new_hand_updates_flags(self):
        self.player_chips[1] = 0
        self.manager.reset_for_new_hand()
        assert self.bets == [0, 0, 0]
        assert self.totals == [0, 0, 0]
        assert self.active == [True, False, False]

    def test_post_blind_clamps_to_stack(self):
        contribution = self.manager.post_blind(1, 100)
        assert contribution == 50
        assert self.player_chips[1] == 0
        assert self.bets[1] == 50
        assert self.totals[1] == 50

    def test_apply_bet_all_in_flag(self):
        contribution, fully_applied = self.manager.apply_bet(1, 80)
        assert contribution == 50
        assert fully_applied is False
        assert self.player_chips[1] == 0
        assert self.bets[1] == 50

    def test_next_player_with_chips(self):
        self.active[1] = False
        # Seat 0 should be returned regardless of include_start because seat 1 is inactive
        assert self.manager.next_player_with_chips(0) == 0
        assert self.manager.next_player_with_chips(0, include_start=False) == 0
        assert self.manager.next_player_with_chips(0, include_start=True) == 0
        # Starting from seat 1 (inactive) or 2 (no chips) should still wrap to seat 0
        assert self.manager.next_player_with_chips(1) == 0
        assert self.manager.next_player_with_chips(2) == 0


class TestBettingRoundTracker:
    def test_reset_and_note_raise(self):
        tracker = BettingRoundTracker()
        tracker.current_bet = 20
        tracker.previous_raise_amount = 10
        tracker.actions_this_round = 2
        tracker.last_raiser = 1
        tracker.reset_round()
        assert tracker.current_bet == 0
        assert tracker.previous_raise_amount == 0
        assert tracker.actions_this_round == 0
        assert tracker.last_raiser is None

        tracker.note_raise(raise_to=60, previous_bet=20)
        assert tracker.current_bet == 60
        assert tracker.previous_raise_amount == 40

    def test_clone_creates_independent_copy(self):
        tracker = BettingRoundTracker(current_bet=40, previous_raise_amount=20, actions_this_round=3, last_raiser=2)
        clone = tracker.clone()
        assert clone is not tracker
        assert clone.current_bet == tracker.current_bet
        tracker.current_bet = 10
        assert clone.current_bet == 40
