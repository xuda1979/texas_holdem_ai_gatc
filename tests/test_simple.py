"""Smoke tests covering the top-level modular architecture."""

from __future__ import annotations

from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.gui.playStrategy import HumanStrategy, RandomAIStrategy
from poker_ai.systems import Registry, RulesSubsystem


def test_strategy_instantiation() -> None:
    """Human and random AI strategies should expose the expected properties."""

    human = HumanStrategy()
    ai = RandomAIStrategy()

    assert human.is_human is True
    assert ai.is_human is False


def test_rules_subsystem_creates_games() -> None:
    """Rules subsystem should create games with the requested configuration."""

    rules = RulesSubsystem.create(num_players=3, starting_stack=750, verbose=False)

    game = rules.new_game(cash_config={"enabled": False})

    assert game.rules.num_players == 3
    assert game.rules.starting_stack == 750
    assert game.rules.player_chips == [750, 750, 750]


def test_registry_tracks_subsystems() -> None:
    """The registry should record subsystem metadata for observability."""

    registry = Registry()
    rules = RulesSubsystem.create(num_players=2, starting_stack=500, verbose=False)
    registry.register(rules)

    assert registry.get("rules") is rules
    assert registry.summary() == [
        {
            "name": "rules",
            "component": "TexasHoldemRules",
            "dependencies": [],
        }
    ]


def test_engine_initialises_with_strategies() -> None:
    """TexasHoldem should accept a mixture of human and AI strategies."""

    strategies = [HumanStrategy(), RandomAIStrategy(), RandomAIStrategy()]

    game = TexasHoldem(
        num_players=len(strategies),
        starting_stack=1_000,
        player_strategies=strategies,
        verbose=False,
        cash_config={"enabled": False},
    )

    assert len(game.player_strategies) == len(strategies)
    assert game.rules.player_chips == [1_000, 1_000, 1_000]
    assert game.rules.small_blind == 10
    assert game.rules.big_blind == 20


def test_engine_handles_all_in() -> None:
    """Game engine should correctly handle an all-in scenario."""

    strategies = [RandomAIStrategy(), RandomAIStrategy()]
    game = TexasHoldem(
        num_players=2,
        starting_stack=100,
        player_strategies=strategies,
        verbose=False,
        cash_config={"enabled": False},
    )
    game.initialize_game()

    # Player 0 (SB) goes all-in
    game.process_action(game.rules.current_player, "raise", 100)

    # Advance to next player
    game.rules.advance_turn()

    # Player 1 (BB) calls
    game.process_action(game.rules.current_player, "call")

    assert game.rules.player_chips == [0, 0]
    assert game.rules.pot == 200
