from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayerState:
    stack: int
    committed: int = 0
    has_folded: bool = False
    is_all_in: bool = False


@dataclass
class GameState:
    big_blind: int
    dealer_index: int
    street: str  # "preflop","flop","turn","river"
    current_bet_to: int = 0
    last_raise_size: int = 0
    last_aggressor: int | None = None
    players: list[PlayerState] = field(default_factory=list)
    pots: list[tuple[int, tuple[int, ...]]] = field(
        default_factory=list
    )  # (amount, eligible)

    def reset_street(self, street: str) -> None:
        self.street = street
        self.current_bet_to = 0
        self.last_raise_size = 0
        self.last_aggressor = None
        for p in self.players:
            p.committed = 0
