from utils.action_mapping import get_action_from_index

# Mock GameState for testing/demo purposes
class MockGameState:
    def __init__(self, pot, current_bet):
        self.pot = pot
        self.current_bet = current_bet


def run_demo():
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

    gs_small_stack_scenario = MockGameState(pot=100, current_bet=10)
    small_player_stack = 30
    print("\n--- Testing all-in and stack limit (pot=100, current_bet=10, player_stack=30) ---")
    print(f"Index 6 (Raise 100% pot): {get_action_from_index(6, gs_small_stack_scenario, small_player_stack)}")
    print(f"Index 9 (All-in): {get_action_from_index(9, gs_small_stack_scenario, small_player_stack)}")

    gs_call_test = MockGameState(pot=100, current_bet=75)
    print("\n--- Testing call amount (pot=100, current_bet=75, player_stack=200) ---")
    print(f"Index 2 (Call): {get_action_from_index(2, gs_call_test, player_stack_size)}")

    gs_rounding_test = MockGameState(pot=101, current_bet=0)
    print("\n--- Testing rounding (pot=101, current_bet=0, player_stack=200) ---")
    print(f"Index 3 (Raise 25% pot): {get_action_from_index(3, gs_rounding_test, player_stack_size)}")
    print(f"Index 4 (Raise 50% pot): {get_action_from_index(4, gs_rounding_test, player_stack_size)}")

    gs_raise_less_than_current = MockGameState(pot=50, current_bet=20)
    print("\n--- Testing raise amount less than current bet (pot=50, current_bet=20, player_stack=200) ---")
    print(f"Index 3 (Raise 25% pot): {get_action_from_index(3, gs_raise_less_than_current, player_stack_size)}")

    gs_zero_pot = MockGameState(pot=0, current_bet=0)
    print("\n--- Testing with zero pot (pot=0, current_bet=0, player_stack=200) ---")
    for i in range(3, 9):
        action, val = get_action_from_index(i, gs_zero_pot, player_stack_size)
        print(f"Index {i}: Action: {action}, Amount: {val}")

    gs_small_pot = MockGameState(pot=1, current_bet=0)
    print("\n--- Testing with very small pot (pot=1, current_bet=0, player_stack=200) ---")
    print(f"Index 3 (Raise 25% pot): {get_action_from_index(3, gs_small_pot, player_stack_size)}")

    gs_check_illegal = MockGameState(pot=100, current_bet=20)
    print("\n--- Testing check when current_bet > 0 (pot=100, current_bet=20, player_stack=200) ---")
    print(f"Index 1 (Check): {get_action_from_index(1, gs_check_illegal, player_stack_size)}")


if __name__ == "__main__":
    run_demo()
