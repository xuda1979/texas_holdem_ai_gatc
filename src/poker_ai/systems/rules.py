"""Subsystem for poker rules and game setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.rules.texas_holdem_rules import TexasHoldemRules

from .base import Subsystem


@dataclass
class RulesSubsystem(Subsystem[TexasHoldemRules]):
    """Provide helpers for working with :class:`TexasHoldemRules`."""

    @classmethod
    def create(
        cls,
        *,
        num_players: int = 2,
        starting_stack: int = 1000,
        verbose: bool = False,
    ) -> "RulesSubsystem":
        rules = TexasHoldemRules(num_players=num_players, starting_stack=starting_stack, verbose=verbose)
        return cls(name="rules", component=rules)

    def new_game(
        self,
        num_players: int | None = None,
        starting_stack: int | None = None,
        **kwargs: Any,
    ) -> TexasHoldem:
        players = num_players or self.component.num_players
        stack = starting_stack or self.component.starting_stack
        if "verbose" not in kwargs:
            kwargs["verbose"] = self.component.verbose
        game = TexasHoldem(num_players=players, starting_stack=stack, **kwargs)
        return game
