"""
Maps action indices to game actions and amounts.
"""
from game_engine.game_state import GameState

def get_action_from_index(action_index: int, game_state: GameState, player_stack: int) -> tuple[str, int | None]:
    """
    Converts an action index (0-9) to a game action string and amount.

    Args:
        action_index: An integer from 0 to 9 representing the action.
        game_state: The current game state object.
        player_stack: The current stack size of the player.

    Returns:
        A tuple containing the action string (e.g., "fold", "raise")
        and the amount for the action (or None if not applicable).
    """
    action_string = ""
    amount = None

    if action_index == 0:
        action_string = "fold"
        amount = None
    elif action_index == 1:
        action_string = "check"
        amount = None
        # Note: Legality of check (e.g. if current_bet > 0) will be handled by the game engine.
    elif action_index == 2:
        action_string = "call"
        # Assuming player_current_bet is what the player has already put in the pot in the current round.
        # For simplicity now, to_call is just game_state.current_bet.
        # This implies the player needs to put in game_state.current_bet to call.
        # A more accurate calculation would be:
        # player_bet_in_round = game_state.get_player_bet_in_round(player_id) # Needs implementation in GameState
        # to_call = game_state.current_bet - player_bet_in_round
        # For now, using the simplified version:
        to_call = game_state.current_bet
        amount = int(round(to_call))
    elif action_index == 3:
        action_string = "raise"
        # Bet/Raise 25% of Pot
        # Amount is the total bet amount.
        raise_amount = game_state.pot * 0.25
        amount = int(round(raise_amount))
    elif action_index == 4:
        action_string = "raise"
        # Bet/Raise 50% of Pot
        raise_amount = game_state.pot * 0.50
        amount = int(round(raise_amount))
    elif action_index == 5:
        action_string = "raise"
        # Bet/Raise 75% of Pot
        raise_amount = game_state.pot * 0.75
        amount = int(round(raise_amount))
    elif action_index == 6:
        action_string = "raise"
        # Bet/Raise 100% of Pot
        raise_amount = game_state.pot * 1.0
        amount = int(round(raise_amount))
    elif action_index == 7:
        action_string = "raise"
        # Bet/Raise 150% of Pot
        raise_amount = game_state.pot * 1.5
        amount = int(round(raise_amount))
    elif action_index == 8:
        action_string = "raise"
        # Bet/Raise 200% of Pot
        raise_amount = game_state.pot * 2.0
        amount = int(round(raise_amount))
    elif action_index == 9:
        action_string = "raise" # Or "bet" if current_bet is 0, game engine can alias.
        amount = player_stack # All-in
    else:
        raise ValueError(f"Invalid action_index: {action_index}. Must be 0-9.")

    # Ensure the bet/raise amount is at least the current bet if it's a raise,
    # or the minimum bet if opening.
    # Also, cap by player_stack.
    # These complexities will be handled by the game engine or in a later refinement.
    # For now, the mapping is direct. If amount is not None, it should be an int.
    if amount is not None:
        amount = int(round(amount))
        # A simple clamp to not bet more than stack.
        # Game engine will have more robust logic for this (e.g. side pots).
        if amount > player_stack:
            amount = player_stack
        # If it's a raise, the amount should be at least the current bet plus a minimum raise unit,
        # or if it's an opening bet, at least a minimum bet.
        # For now, if current_bet > 0 and the calculated amount for "raise" is less than current_bet,
        # it's not a valid raise. The game engine should handle this.
        # The current logic simply calculates the target bet amount for "raise" actions.

    return action_string, amount

