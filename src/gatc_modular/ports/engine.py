from __future__ import annotations

from dataclasses import dataclass
import sys

if sys.version_info >= (3, 11):
    from typing import Any, Dict, List, Mapping, Optional, Protocol, Self, runtime_checkable
else:
    from typing import Any, Dict, List, Mapping, Optional, Protocol, runtime_checkable
    from typing import TypeVar as _TypeVar

    Self = _TypeVar("Self", bound="Engine")  # type: ignore[misc,assignment]

PlayerId = int
Action = int
Observation = Any


@dataclass(frozen=True)
class StepResult:
    """The result of applying one action to the engine."""
    observation: Observation
    reward: Mapping[PlayerId, float]
    done: bool
    info: Dict[str, Any]


@runtime_checkable
class Engine(Protocol):
    """Minimal protocol any poker engine should satisfy to interoperate with services."""
    num_players: int

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the initial observation."""
        ...

    def current_player(self) -> PlayerId:
        """Return the id of the player whose turn it is."""
        ...

    def legal_actions(self) -> List[Action]:
        """Return the list of legal discrete actions at the current state."""
        ...

    def step(self, action: Action) -> StepResult:
        """Apply `action`, advance the environment, and return the transition result."""
        ...

    def is_terminal(self) -> bool:
        """True if the game is over."""
        ...

    def winner(self) -> Optional[PlayerId]:
        """Return the winner's player id if terminal and there is a winner; otherwise None."""
        ...

    def clone(self) -> Self:
        """Return a (cheap) copy of the engine (useful for search/self-play)."""
        ...
