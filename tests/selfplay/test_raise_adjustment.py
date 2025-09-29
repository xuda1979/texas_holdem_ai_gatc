from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.selfplay.self_play import SelfPlay
from poker_ai.utils.action_mapping import get_action_from_index


class _DummyTrainer:
    def __init__(self):
        self.config = {}


def test_apply_action_uses_raise_increment():
    trainer = _DummyTrainer()
    sp = SelfPlay(trainer, {"num_players": 2, "starting_stack": 1000})

    game = TexasHoldem(num_players=2, starting_stack=1000, verbose=False)
    game.initialize_game()

    current_player = game.rules.current_player
    initial_bets = list(game.rules.bets)
    action_index = 2  # first raise bucket beyond fold/call
    action = get_action_from_index(action_index, game, player_id=current_player)
    target_total = action.amount_to

    assert target_total is not None and target_total > game.rules.current_bet

    sp._apply_action_in_place(game, current_player, action_index)

    assert game.rules.bets[current_player] == target_total
    assert game.rules.current_bet == target_total
    # ensure the opposing player's bet has not unintentionally changed
    other_player = 1 - current_player
    assert game.rules.bets[other_player] == initial_bets[other_player]
