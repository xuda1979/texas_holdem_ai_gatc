import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for path in (PROJECT_ROOT, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from gatc_holdem.engine import CashTable  # noqa: E402


def _setup_table():
    table = CashTable(small_blind=1, big_blind=2, rake_pct=0.05, rake_cap=5)
    table.seat_player(pid=0, bankroll=500, buyin_bb=100)
    table.seat_player(pid=1, bankroll=300, buyin_bb=50)
    table.seat_player(pid=2, bankroll=300, buyin_bb=40)
    table.start_hand([0, 1, 2])
    return table


def test_start_hand_records_initial_stacks():
    table = _setup_table()
    # Active players should have their starting stack captured
    assert table.stack_at_hand_start(0) == table.players[0].stack
    assert table.stack_at_hand_start(1) == table.players[1].stack
    assert table.stack_at_hand_start(2) == table.players[2].stack

    # Remove player 2 for the next hand and ensure snapshot updates
    table.players[2].stack = 0
    table.start_hand([0, 1])
    assert table.stack_at_hand_start(0) == table.players[0].stack
    assert table.stack_at_hand_start(1) == table.players[1].stack
    assert table.stack_at_hand_start(2) == 0


def test_short_blind_all_in_sets_all_in_flag():
    table = _setup_table()
    # Exhaust player 2 stack to simulate short blind
    table.players[2].stack = 1
    posted = table.post_blind(2, table.big_blind)
    assert posted == 1
    assert table.players[2].stack == 0
    assert table.players[2].all_in is True


def test_legal_actions_min_raise_bounds():
    table = _setup_table()
    actor = table.players[0]
    actor.committed = 0
    table.players[1].commit(2)  # big blind amount already in
    # Player 0 faces a raise to 6 with last full raise size of 4
    actions = table.legal_actions(pid=0, to_call=6, last_raise=4)
    assert actions.call_amount == 6
    assert actions.raise_bounds is not None
    assert actions.raise_bounds.min_total == 10
    assert actions.raise_bounds.max_total == actor.stack


def test_short_stack_cannot_reopen_action():
    table = _setup_table()
    player = table.players[1]
    player.stack = 3
    player.committed = 0
    actions = table.legal_actions(pid=1, to_call=2, last_raise=2)
    assert actions.call_amount == 2
    assert actions.raise_bounds is not None
    assert actions.raise_bounds.min_total == actions.raise_bounds.max_total == 3
    assert actions.raise_bounds.min_reopens is False


def test_build_side_pots_with_folded_players():
    table = _setup_table()
    table.players[0].commit(50)
    table.players[1].commit(100)
    table.players[2].stack = 200
    table.players[2].commit(100)
    table.players[2].folded = True
    table.build_side_pots()
    assert len(table.pot_layers) == 2
    main_pot, side_pot = table.pot_layers
    assert main_pot.amount == 150
    assert set(main_pot.eligible) == {0, 1}
    assert side_pot.amount == 100
    assert set(side_pot.eligible) == {1}


def test_take_rake_no_flop_no_drop():
    table = _setup_table()
    table.players[0].commit(50)
    table.players[1].commit(50)
    table.build_side_pots()
    rake = table.take_rake()
    assert rake == 0
    table.mark_flop_seen()
    rake = table.take_rake()
    assert rake == 5  # 5% of 100 capped at 5
    assert abs(table.total_pot - 95) < 1e-9


def test_pay_out_uses_odd_chip_rule():
    table = _setup_table()
    table.button_idx = 0
    table.players[0].commit(50)
    table.players[1].commit(50)
    table.build_side_pots()
    table.pot_layers[0].amount = 101
    table.pay_out({0: [0, 1]})
    assert table.players[0].stack == table.big_blind * 100
    assert table.players[1].stack == table.big_blind * 50 + 1


def test_top_up_respects_bankroll():
    table = _setup_table()
    table.players[1].stack = 50
    table.players[1].bankroll = 40
    added = table.top_up(1, target_bb=60)
    assert added == 40
    assert table.players[1].stack == 90
    assert table.players[1].bankroll == 0


def test_seat_player_random_defaults_within_range():
    table = CashTable(
        small_blind=1,
        big_blind=2,
        min_buyin_bb=40,
        max_buyin_bb=100,
        min_bankroll_buyins=2,
        max_bankroll_buyins=4,
    )
    random.seed(0)
    table.seat_player(pid=7)
    player = table.players[7]
    assert table.min_buyin_bb <= player.stack / table.big_blind <= table.max_buyin_bb
    total_funds = player.stack + player.bankroll
    bankroll_buyins = total_funds / player.stack
    assert table.min_bankroll_buyins <= bankroll_buyins <= table.max_bankroll_buyins
    assert bankroll_buyins == 3
    assert player.stack == 94 * table.big_blind
    assert player.bankroll == player.stack * (bankroll_buyins - 1)

