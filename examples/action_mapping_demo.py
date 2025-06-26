"""Demonstration of utils.action_mapping.get_action_from_index."""
from utils.action_mapping import get_action_from_index


# Mock GameState for testing
class MockGameState:
    def __init__(self, pot, current_bet):
        self.pot = pot
        self.current_bet = current_bet
        # self.player_current_bet_in_round = 0  # Example, if needed for more complex call logic

# Test cases
if __name__ == "__main__":
    gs_no_bet = MockGameState(pot=100, current_bet=0)
    gs_bet_exists = MockGameState(pot=150, current_bet=50)
    player_stack_size = 200

    actions_to_test = range(10)

    print("--- Testing with no current bet (pot=100, current_bet=0, player_stack=200) ---")
    for i in actions_to_test:
        action, val = get_action_from_index(i, gs_no_bet, player_stack_size)
        print(f"Index {i}: Action: {action}, Amount: {val}")

    print("\n--- Testing with existing bet (pot=150, current_bet=50, player_stack=200) ---")
    for i in actions_to_test:
        action, val = get_action_from_index(i, gs_bet_exists, player_stack_size)
        print(f"Index {i}: Action: {action}, Amount: {val}")

    # Test all-in scenario where pot-based raise exceeds stack
    gs_small_stack_scenario = MockGameState(pot=100, current_bet=10)
    small_player_stack = 30
    print("\n--- Testing all-in and stack limit (pot=100, current_bet=10, player_stack=30) ---")
    print(f"Index 6 (Raise 100% pot): {get_action_from_index(6, gs_small_stack_scenario, small_player_stack)}")  # Pot = 100, but capped at 30
    print(f"Index 9 (All-in): {get_action_from_index(9, gs_small_stack_scenario, small_player_stack)}")  # All-in = 30

    # Test call amount
    gs_call_test = MockGameState(pot=100, current_bet=75)
    print("\n--- Testing call amount (pot=100, current_bet=75, player_stack=200) ---")
    print(f"Index 2 (Call): {get_action_from_index(2, gs_call_test, player_stack_size)}")  # Call 75

    # Test rounding
    gs_rounding_test = MockGameState(pot=101, current_bet=0)
    print("\n--- Testing rounding (pot=101, current_bet=0, player_stack=200) ---")
    print(f"Index 3 (Raise 25% pot): {get_action_from_index(3, gs_rounding_test, player_stack_size)}")  # 101 * 0.25 = 25.25 -> 25
    print(f"Index 4 (Raise 50% pot): {get_action_from_index(4, gs_rounding_test, player_stack_size)}")  # 101 * 0.50 = 50.5 -> 51

    # Test case where raise amount is less than current bet (game engine should handle this)
    # For action_index 3 (25% pot raise): pot = 50, current_bet = 20. Raise amount = 50 * 0.25 = 12.5 -> 13.
    # This is not a valid raise if current_bet is 20. The function currently returns (raise, 13).
    # The game engine will need to validate this and adjust or declare it illegal.
    gs_raise_less_than_current = MockGameState(pot=50, current_bet=20)
    print("\n--- Testing raise amount less than current bet (pot=50, current_bet=20, player_stack=200) ---")
    print(f"Index 3 (Raise 25% pot): {get_action_from_index(3, gs_raise_less_than_current, player_stack_size)}")

    # Test case where calculated raise amount is 0 (e.g. pot is 0 or very small)
    gs_zero_pot = MockGameState(pot=0, current_bet=0)
    print("\n--- Testing with zero pot (pot=0, current_bet=0, player_stack=200) ---")
    for i in range(3, 9):  # Raise actions
        action, val = get_action_from_index(i, gs_zero_pot, player_stack_size)
        print(f"Index {i}: Action: {action}, Amount: {val}")  # Should be (raise, 0)

    gs_small_pot = MockGameState(pot=1, current_bet=0)
    print("\n--- Testing with very small pot (pot=1, current_bet=0, player_stack=200) ---")
    print(f"Index 3 (Raise 25% pot): {get_action_from_index(3, gs_small_pot, player_stack_size)}")  # 1 * 0.25 = 0.25 -> 0. (raise, 0)

    # Test action_index 1 (check) when current_bet > 0
    gs_check_illegal = MockGameState(pot=100, current_bet=20)
    print("\n--- Testing check when current_bet > 0 (pot=100, current_bet=20, player_stack=200) ---")
    print(f"Index 1 (Check): {get_action_from_index(1, gs_check_illegal, player_stack_size)}")  # ("check", None) - Game engine handles legality
