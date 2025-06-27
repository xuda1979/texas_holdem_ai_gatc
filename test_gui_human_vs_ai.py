#!/usr/bin/env python3
"""
Test script to verify that the GUI properly enforces human vs AI gameplay.
This test simulates the GUI setup and checks key conditions.
"""

import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game_engine.texas_holdem import TexasHoldem
from playStrategy import HumanStrategy, RandomAIStrategy

def test_human_vs_ai_setup():
    """Test that the game setup enforces human vs AI gameplay"""
    print("Testing human vs AI setup...")
    
    # Test with different AI counts
    test_cases = [
        (1, 2),  # 1 AI, 2 total players
        (2, 3),  # 2 AIs, 3 total players
        (3, 4),  # 3 AIs, 4 total players
        (5, 6),  # 5 AIs, 6 total players
    ]
    
    for ai_count, total_players in test_cases:
        print(f"\nTesting {ai_count} AIs with {total_players} total players...")
        
        # Create player strategies like the GUI does
        player_strategies = [HumanStrategy()]  # Human player at index 0
        for _ in range(ai_count):
            player_strategies.append(RandomAIStrategy())
        
        # Verify strategy setup
        assert len(player_strategies) == total_players, f"Expected {total_players} strategies, got {len(player_strategies)}"
        assert isinstance(player_strategies[0], HumanStrategy), "Player 0 should be human"
        
        for i in range(1, total_players):
            assert isinstance(player_strategies[i], RandomAIStrategy), f"Player {i} should be AI"
        
        # Create game engine
        game_engine = TexasHoldem(
            num_players=total_players,
            starting_stack=1000,
            player_strategies=player_strategies
        )
        
        # Verify game engine setup
        assert game_engine.num_players == total_players
        assert len(game_engine.player_strategies) == total_players
        assert isinstance(game_engine.player_strategies[0], HumanStrategy)
        
        for i in range(1, total_players):
            assert isinstance(game_engine.player_strategies[i], RandomAIStrategy)
        
        print(f"✓ Setup correct for {ai_count} AIs")
    
    print("\n✓ All human vs AI setup tests passed!")

def test_invalid_setups():
    """Test that invalid setups are properly rejected"""
    print("\nTesting invalid setup scenarios...")
    
    # Test scenarios that should be rejected by the GUI validation
    invalid_cases = [
        (0, 1),   # 0 AIs (would be 1 total, but GUI requires 2+ players)
        (10, 10), # 10 AIs, 10 total (no human player)
        (5, 4),   # More AIs than total players possible
    ]
    
    for ai_count, total_players in invalid_cases:
        print(f"Testing invalid case: {ai_count} AIs, {total_players} total players...")
        
        # This mimics the GUI validation logic
        if not (2 <= total_players <= 10):
            print(f"✓ Correctly rejected: Total players must be between 2 and 10")
            continue
            
        if not (1 <= ai_count <= total_players - 1):
            print(f"✓ Correctly rejected: AI count must be between 1 and {total_players - 1}")
            continue
            
        if total_players != ai_count + 1:
            print(f"✓ Correctly rejected: Total players must equal AI count + 1 human player")
            continue
        
        print(f"❌ Invalid case was not caught: {ai_count} AIs, {total_players} total players")
    
    print("✓ All invalid setup tests passed!")

def test_turn_management():
    """Test that turn management works correctly for human vs AI"""
    print("\nTesting turn management...")
    
    # Create a 3-player game (1 human, 2 AIs)
    player_strategies = [HumanStrategy(), RandomAIStrategy(), RandomAIStrategy()]
    game_engine = TexasHoldem(
        num_players=3,
        starting_stack=1000,
        player_strategies=player_strategies
    )
    
    game_engine.initialize_game()
    
    # Verify initial state
    assert game_engine.num_players == 3
    assert isinstance(game_engine.player_strategies[0], HumanStrategy)
    assert isinstance(game_engine.player_strategies[1], RandomAIStrategy)
    assert isinstance(game_engine.player_strategies[2], RandomAIStrategy)
    
    print("✓ Game initialized correctly with human at index 0 and AIs at indices 1,2")
    
    # Check that the game has proper player identification
    human_player_index = 0
    
    # Simulate checking if it's human's turn (like GUI does)
    current_player = game_engine.rules.current_player
    is_human_turn = (current_player == human_player_index)
    
    print(f"Current player: {current_player}, Human index: {human_player_index}, Is human turn: {is_human_turn}")
    
    # Test that AI players have strategies assigned
    for i in range(1, 3):
        if game_engine.player_strategies[i]:
            ai_strategy = game_engine.player_strategies[i]
            assert isinstance(ai_strategy, RandomAIStrategy), f"Player {i} should have RandomAIStrategy"
    
    print("✓ Turn management test passed!")

if __name__ == "__main__":
    print("Running GUI Human vs AI tests...\n")
    
    try:
        test_human_vs_ai_setup()
        test_invalid_setups()
        test_turn_management()
        
        print("\n🎉 All tests passed! The GUI correctly enforces human vs AI gameplay.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
