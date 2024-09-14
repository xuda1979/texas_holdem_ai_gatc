# playStrategy.py

import random

class PlayerStrategy:
    @property
    def is_human(self):
        raise NotImplementedError

    def choose_action(self, game, player_index):
        raise NotImplementedError


class RandomAIStrategy(PlayerStrategy):
    @property
    def is_human(self):
        return False

    def choose_action(self, game, player_index):
        # Determine the amount needed to call
        amount_to_call = game.rules.current_bet - game.rules.bets[player_index]

        # Determine valid actions based on the current bet
        if amount_to_call > 0:
            if game.rules.player_chips[player_index] < amount_to_call:
                # Player can only call (go all-in) or fold
                actions = ['call', 'fold']
            else:
                # Player can call, raise, or fold
                actions = ['call', 'raise', 'fold']
        else:
            # No current bet; player can check or bet
            actions = ['check', 'bet']

        action = random.choice(actions)

        if action in ['raise', 'bet']:
            min_raise = game.get_min_raise_amount(player_index)
            max_raise = game.get_max_raise_amount(player_index)
            if max_raise < min_raise:
                if amount_to_call > 0:
                    return 'call', None  # Can't raise; must call or fold
                else:
                    return 'check', None  # Can't raise; must check
            # AI decides on a raise amount within the allowed range
            raise_amount = random.randint(min_raise, min(max_raise, min_raise + 100))
            return action, raise_amount
        return action, None


class HumanStrategy(PlayerStrategy):
    @property
    def is_human(self):
        return True

    def choose_action(self, game, player_index):
        while True:
            print(f"\n--- Player {player_index + 1}'s Turn ---")
            print(f"Current pot: {game.rules.pot} chips")
            print(f"Your chips: {game.rules.player_chips[player_index]} chips")
            amount_to_call = game.rules.current_bet - game.rules.bets[player_index]
            if amount_to_call > 0:
                print(f"Amount to call: {amount_to_call} chips")
            else:
                print("You can check.")

            if amount_to_call > 0:
                valid_actions = ['call', 'raise', 'fold']
            else:
                valid_actions = ['check', 'bet']

            action = input(f"Choose your action ({', '.join(valid_actions)}): ").lower()
            if action in ['raise', 'bet']:
                min_raise = game.get_min_raise_amount(player_index)
                max_raise = game.get_max_raise_amount(player_index)
                if max_raise < min_raise:
                    if amount_to_call > 0:
                        print("You don't have enough chips to raise. You can only call or fold.")
                    else:
                        print("You don't have enough chips to raise. You can only check.")
                    continue
                while True:
                    try:
                        raise_amount = int(input(f"Enter raise amount (minimum {min_raise} chips): "))
                        if raise_amount < min_raise:
                            print(f"Raise amount must be at least {min_raise} chips.")
                        elif raise_amount > max_raise:
                            print(f"Raise amount cannot exceed your available chips ({max_raise} chips).")
                        else:
                            break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")
                return action, raise_amount
            elif action in ['call', 'fold', 'check']:
                return action, None
            else:
                print("Invalid action. Please try again.")
