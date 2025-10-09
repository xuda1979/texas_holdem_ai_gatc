from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


ActionDetails = Tuple[str, Optional[int]]


@dataclass
class GameState:
    """In-memory representation of a single Texas Hold'em hand.

    The previous implementation stored mutable attributes directly on the
    instance.  Converting to a dataclass provides clear defaults, type hints and
    makes it easier to reason about invariants when the engine mutates the
    state during game play.
    """

    pot: int = 0
    community_cards: List[str] = field(default_factory=list)
    player_hands: MutableMapping[Any, Sequence[str]] = field(default_factory=dict)
    betting_history: List[Tuple[Any, ActionDetails]] = field(default_factory=list)
    current_bet: int = 0
    current_round: str = "pre-flop"
    players: List[Any] = field(default_factory=list)
    player_order: List[str] = field(default_factory=list)
    # Some utilities expect `betting_round` attribute to mirror `current_round`.
    betting_round: str = "pre-flop"

    def __post_init__(self) -> None:
        # Ensure `betting_round` always starts in sync with `current_round`.
        self.betting_round = self.current_round

    def set_players(self, players: Iterable[Any]) -> None:
        self.players = list(players)
        # Assume players is a list of objects with a player_id attribute
        try:
            self.player_order = [str(p.player_id) for p in self.players]
        except AttributeError:
            # Fallback if players are provided as IDs or lacking attribute
            self.player_order = [str(p) for p in self.players]

    def get_player(self, player_id: Any) -> Any:
        """Return player object by id if available."""
        for p in self.players:
            if getattr(p, "player_id", None) == player_id or str(p) == str(player_id):
                return p
        return None

    def set_player_hand(self, player_id: Any, hand: Sequence[str]) -> None:
        self.player_hands[player_id] = hand

    def add_community_cards(self, cards: Iterable[str]) -> None:
        self.community_cards.extend(cards)

    def record_action(self, player_id: Any, action_details: ActionDetails) -> None:
        """
        Records a player's action in the betting history.

        Args:
            player_id: The ID of the player performing the action.
            action_details: A tuple representing the action.
                Expected format: (action_name_str, amount_int_or_None)
                Examples:
                    ('fold', None)
                    ('check', None)
                    ('call', 100)
                    ('bet', 200)
                    ('raise', 500)
        """
        if not isinstance(action_details, tuple) or len(action_details) != 2:
            raise ValueError("action_details must be a (action, amount) tuple")
        action, amount = action_details
        if not isinstance(action, str):
            raise TypeError("action must be a string")
        if amount is not None and not isinstance(amount, int):
            raise TypeError("amount must be an integer or None")
        self.betting_history.append((player_id, action_details))

    def set_current_bet(self, bet: int) -> None:
        if bet < 0:
            raise ValueError("bet must be non-negative")
        self.current_bet = bet

    def reset(self, keep_players: bool = True) -> None:
        """Reset the mutable state for a new hand.

        Args:
            keep_players: When ``True`` the player list/order is preserved.  Set
                to ``False`` to remove previously registered players entirely.
        """

        self.pot = 0
        self.community_cards = []
        self.player_hands = {}
        self.betting_history = []
        self.current_bet = 0
        self.current_round = "pre-flop"
        self.betting_round = "pre-flop"

        if keep_players:
            self.player_order = [
                str(getattr(player, "player_id", player)) for player in self.players
            ]
        else:
            self.players = []
            self.player_order = []

    def to_dict(self, include_players: bool = True) -> Dict[str, Any]:
        """Serialize the state to a JSON-friendly dictionary."""

        data: Dict[str, Any] = {
            "pot": self.pot,
            "community_cards": list(self.community_cards),
            "player_hands": {
                str(pid): list(cards) for pid, cards in self.player_hands.items()
            },
            "betting_history": [
                {
                    "player_id": str(player_id),
                    "action": action,
                    "amount": amount,
                }
                for player_id, (action, amount) in self.betting_history
            ],
            "current_bet": self.current_bet,
            "current_round": self.current_round,
            "betting_round": self.betting_round,
            "player_order": list(self.player_order),
        }

        if include_players:
            data["players"] = [
                getattr(player, "player_id", str(player)) for player in self.players
            ]

        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GameState":
        """Rehydrate a :class:`GameState` from :meth:`to_dict` output."""

        state = cls(
            pot=payload.get("pot", 0),
            community_cards=list(payload.get("community_cards", [])),
            current_bet=payload.get("current_bet", 0),
            current_round=payload.get("current_round", "pre-flop"),
        )

        state.betting_round = payload.get("betting_round", state.current_round)
        state.player_order = list(payload.get("player_order", []))

        hands = payload.get("player_hands", {})
        state.player_hands = {pid: tuple(cards) for pid, cards in hands.items()}

        history = []
        for item in payload.get("betting_history", []):
            history.append(
                (
                    item.get("player_id"),
                    (item.get("action", ""), item.get("amount")),
                )
            )
        state.betting_history = history

        if "players" in payload:
            state.players = list(payload["players"])

        return state
