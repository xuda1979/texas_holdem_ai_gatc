#!/usr/bin/env python3
"""
Test script to verify deck management and hand transitions work correctly
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_engine.texas_holdem import TexasHoldem


def test_multiple_hands():
    """Test that we can play multiple hands without deck exhaustion"""
    print("Testing multiple hands...")  # Create game with 2 players
    game = TexasHoldem(num_players=2, starting_stack=1000)

    for hand_num in range(5):  # Test 5 hands
        print(f"\n--- Hand {hand_num + 1} ---")

        # Initialize new hand
        game.initialize_game()
        print(f"Deck size after initialization: {len(game.rules.deck)}")
        print(f"Player hands: {game.rules.hands}")
        print(f"Community cards: {game.rules.community_cards}")

        # Simulate betting rounds
        # Pre-flop (hands already dealt)
        cards_used = 4  # 2 cards per player
        # Flop
        game.rules.deal_community_cards("flop")
        cards_used += 3 + 1  # 3 community cards + 1 burn card
        print(f"After flop - Deck size: {len(game.rules.deck)}, Cards used: {cards_used}")
        print(f"Community cards: {game.rules.community_cards}")

        # Turn
        game.rules.deal_community_cards("turn")
        cards_used += 1 + 1  # 1 community card + 1 burn card
        print(f"After turn - Deck size: {len(game.rules.deck)}, Cards used: {cards_used}")
        print(f"Community cards: {game.rules.community_cards}")

        # River
        game.rules.deal_community_cards("river")
        cards_used += 1 + 1  # 1 community card + 1 burn card
        print(f"After river - Deck size: {len(game.rules.deck)}, Cards used: {cards_used}")
        print(f"Community cards: {game.rules.community_cards}")
        # Total cards used should be 10 (4 hole + 5 community + 3 burn)
        remaining_cards = 52 - cards_used
        print(f"Expected remaining cards: {remaining_cards}, Actual: {len(game.rules.deck)}")

        assert (
            len(game.rules.deck) == remaining_cards
        ), f"Deck size mismatch: expected {remaining_cards}, got {len(game.rules.deck)}"
        print("✓ Hand completed successfully")


if __name__ == "__main__":
    test_multiple_hands()
    print("\n✓ All tests passed! Deck management is working correctly.")
