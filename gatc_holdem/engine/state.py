from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    last_aggressor: Optional[int] = None
    players: List[PlayerState] = field(default_factory=list)
    pots: List[Tuple[int, Tuple[int, ...]]] = field(
        default_factory=list
    )  # (amount, eligible)

    def reset_street(self, street: str) -> None:
        self.street = street
        self.current_bet_to = 0
        self.last_raise_size = 0
        self.last_aggressor = None
        for p in self.players:
            p.committed = 0
