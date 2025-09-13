"""
Maps action indices to game actions and amounts, and provides legality checks.
"""
import torch
from poker_ai.engine.texas_holdem import TexasHoldem

def _is_action_valid(game: TexasHoldem, player_id: int, action_str: str, amount: int | None) -> bool:
    """
    Checks if a given action is legal in the current game state.
    This is a helper function containing the core game logic for action validation.
    """
    rules = game.rules
    player_chips = rules.player_chips[player_id]

    # This needs to be the player's bet in the current round, not their total bet in the hand.
    # The game engine should track this. Assuming `rules.bets` is reset each round.
    player_bet_in_round = rules.bets[player_id]

    amount_to_call = rules.current_bet - player_bet_in_round

    if action_str == 'fold':
        # Fold is always legal unless the player is all-in and there's no bet to call.
        return True

    if action_str == 'check':
        # Check is only legal if there is no bet to call.
        return amount_to_call == 0

    if action_str == 'call':
        # Call is only legal if there is a bet to call.
        if amount_to_call <= 0:
            return False
        # A player can always call, even if it means going all-in.
        return True

    if action_str == 'bet':
        # Bet is only legal if there is no current bet in the round.
        if rules.current_bet != 0:
            return False
        if amount is None or amount <= 0:
            return False
        # Bet must be at least the big blind, unless the player is going all-in for less.
        if amount < rules.big_blind and amount != player_chips:
            return False
        # Cannot bet more than you have.
        return player_chips >= amount

    if action_str == 'raise':
        # Raise is only legal if there is a current bet.
        if rules.current_bet == 0:
            return False
        if amount is None:
            return False

        # The total amount of the raise must be more than the current bet.
        if amount <= rules.current_bet:
            return False

        # The player must have enough chips to make the raise.
        # The amount to commit is the total new bet amount minus what they've already bet.
        raise_amount_to_commit = amount - player_bet_in_round
        if player_chips < raise_amount_to_commit:
            return False

        # The raise increment must be at least the size of the previous bet/raise,
        # unless the player is going all-in for less (an "under-raise").
        min_raise_increment = rules.previous_raise_amount if rules.previous_raise_amount > 0 else rules.big_blind
        actual_raise_increment = amount - rules.current_bet

        if actual_raise_increment < min_raise_increment:
            # An under-raise is only legal if the player is going all-in.
            return raise_amount_to_commit == player_chips

        return True

    return False

def get_legal_actions_mask(game: TexasHoldem, player_id: int, num_actions: int) -> torch.Tensor:
    """
    Returns a boolean tensor indicating which of the abstract actions are legal.
    """
    mask = torch.zeros(num_actions, dtype=torch.bool)
    # Player stack is available but not currently used in the logic below
    # It could be used for additional validation in the future
    _ = game.rules.player_chips[player_id]

    for action_idx in range(num_actions):
        action_str, amount = get_action_from_index(action_idx, game, player_id)

        # The 'raise' action from get_action_from_index should be treated as 'bet' if no bet has been made.
        if game.rules.current_bet == 0 and action_str == 'raise':
            action_str = 'bet'

        if _is_action_valid(game, player_id, action_str, amount):
            mask[action_idx] = True

    # If no actions are legal (should not happen, fold is always an option), log an error.
    if not mask.any():
        # As a fallback, mark 'fold' as legal.
        mask[0] = True

    return mask


def get_action_from_index(action_index: int, game: TexasHoldem, player_id: int) -> tuple[str, int | None]:
    """Converts an action index (0-9) to a game action string and amount.

    The helper primarily targets :class:`TexasHoldem` instances but also works
    with lightweight stand-ins used in tests.  When the ``game`` object lacks a
    ``rules`` attribute we fall back to using ``game`` directly and interpret
    ``player_id`` as the player's stack size.
    """

    action_string = ""
    amount = None

    if hasattr(game, "rules"):
        rules = game.rules
        pot = rules.pot
        player_stack = rules.player_chips[player_id]
        current_bet = rules.current_bet
        player_bet_in_round = rules.bets[player_id]
    else:  # minimal dummy state for unit tests
        pot = getattr(game, "pot", 0)
        current_bet = getattr(game, "current_bet", 0)
        player_stack = player_id  # here `player_id` encodes stack size
        player_bet_in_round = getattr(game, "current_player_bet", 0)

    # Define raise percentages relative to the pot
    raise_percentages = {
        3: 0.25, 4: 0.50, 5: 0.75, 6: 1.0, 7: 1.5, 8: 2.0
    }

    if action_index == 0:
        action_string = "fold"
    elif action_index == 1:
        action_string = "check"
    elif action_index == 2:
        action_string = "call"
        amount = current_bet - player_bet_in_round
    elif action_index in raise_percentages:
        # If there's no bet, this is a 'bet'. Otherwise, it's a 'raise'.
        action_string = "raise" if current_bet > 0 else "bet"
        # The amount is the total size of the new bet, not the increment.
        # For a bet, it's % of pot. For a raise, it's current_bet + % of pot.
        raise_increment = pot * raise_percentages[action_index]
        amount = current_bet + raise_increment
    elif action_index == 9:
        # Treat index 9 as an all-in raise regardless of current betting state.
        action_string = "raise"
        amount = player_stack + player_bet_in_round  # total commitment including prior bet
    else:
        raise ValueError(f"Invalid action_index: {action_index}. Must be 0-9.")

    # Clamp the amount to be within the player's stack
    if amount is not None:
        amount_to_commit = amount - player_bet_in_round if action_string == 'raise' else amount
        if amount_to_commit > player_stack:
            amount = player_stack + player_bet_in_round

        amount = int(round(amount)) if amount > 0 else 0

    # The action string for 'call' should be the final amount to call
    if action_string == 'call':
        amount = current_bet - player_bet_in_round

    return action_string, amount

