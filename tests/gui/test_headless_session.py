import random
from types import SimpleNamespace


class DummyGame:
    def __init__(self, actions):
        self._actions = actions
        self.rules = SimpleNamespace(current_bet=0, bets=[0, 0])

    def get_valid_actions(self, player_index):
        return self._actions

    def get_min_raise_amount(self, player_index):
        return 10

    def get_max_raise_amount(self, player_index):
        return 100


class RandomStrategy:
    def choose_action(self, game, player_index):
        action = random.choice(game.get_valid_actions(player_index))
        if action in ["raise", "bet"]:
            return action, game.get_min_raise_amount(player_index)
        return action, None


def test_cli_and_gui_decision_parity_on_recording():
    actions = ["fold", "call", "raise"]
    game = DummyGame(actions)
    cli_strategy = RandomStrategy()
    gui_strategy = RandomStrategy()

    seed = 999
    random.seed(seed)
    cli_action = cli_strategy.choose_action(game, 0)
    random.seed(seed)
    gui_action = gui_strategy.choose_action(game, 0)

    assert cli_action == gui_action
    assert cli_action[0] in actions
