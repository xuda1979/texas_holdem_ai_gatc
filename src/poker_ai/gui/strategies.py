class PlaceholderAIStrategy:
    def choose_action(self, game_state, player_index):
        """
        Placeholder AI: Always calls if possible, otherwise checks.
        Never bets or raises. Folds if no other option.
        """
        # game_state is expected to be an instance of TexasHoldem (the main game class)
        rules = game_state.rules
        current_bet = rules.current_bet
        player_bet = rules.bets[player_index]

        can_check = current_bet == player_bet

        if can_check:
            # print(f"AI (Player {player_index+1}) checks.")
            return "check", 0  # Amount is irrelevant for check

        # If not able to check, means there's a bet to respond to
        amount_to_call = current_bet - player_bet
        if rules.player_chips[player_index] >= amount_to_call:
            # print(f"AI (Player {player_index+1}) calls.")
            # process_action for 'call' does not need an amount, it calculates it.
            return "call", 0  # Amount is not used by process_action for 'call'
        else:
            # Not enough chips to call the full amount, but can call all-in.
            # print(f"AI (Player {player_index+1}) calls all-in.")
            return "call", 0  # Still a 'call' action, process_action will handle all-in logic.

        # This part should ideally not be reached if AI always calls/checks or folds.
        # However, if rules.player_chips[player_index] < amount_to_call (and not 0 for all-in call)
        # it must fold if it cannot call. The logic above covers all-in call.
        # For simplicity, if it cannot call (even all-in), it should fold.
        # The current logic implies it will always attempt to call (even if it's an all-in).
        # If player_chips is 0, it shouldn't be AI's turn anyway.


class HumanStrategy:
    """
    A marker class for human players. Actions are determined by GUI interactions.
    The choose_action method here will not be called by the game engine's main loop
    if the engine is adapted to pause for human input.
    """

    def choose_action(self, game_state, player_index):
        # This method should ideally not be called if the game loop handles human players.
        # If it is called, it implies an issue with the game flow.
        print("Warning: HumanStrategy.choose_action called. This should be handled by GUI.")
        return "fold", 0  # Default safe action if called unexpectedly
