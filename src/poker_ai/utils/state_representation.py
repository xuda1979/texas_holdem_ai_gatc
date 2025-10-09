"""Utility helpers for turning game state into model-friendly tensors.

This module now includes a small Set Transformer style encoder used to
summarise unordered card sets (the player's private hole cards and the
community cards).  The summaries are returned alongside the sequential
history features so that the main model can fuse them as described in the
project specification in :mod:`texas.tex`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

# Assuming GameState and Player will be importable from these paths
# from game_engine.game_state import GameState
# from game_engine.player import Player


# --- Mock classes for development and testing ---
class MockPlayer:
    def __init__(self, player_id: str, hand: list[str], stack: int) -> None:
        self.player_id = player_id
        self.hand = hand
        self.stack = stack
        self.current_bet_in_round = 0  # Player's current contribution in the betting round


class MockGameState:
    def __init__(
        self,
        players: list[MockPlayer],
        community_cards: list[str],
        pot: int,
        current_bet: int,
        betting_round: str,
        betting_history: list[tuple[str, tuple[str, int | None]]],
        player_order: list[str] | None = None,
    ) -> None:  # player_order can be passed if specific order matters
        self.players_map = {
            p.player_id: p for p in players
        }  # Renamed from self.players to avoid confusion
        if player_order:
            self.player_order = player_order
        else:
            self.player_order = [p.player_id for p in players]  # Default order if not specified

        self.community_cards = community_cards
        self.pot = pot
        self.current_bet = current_bet
        self.betting_round = betting_round
        self.betting_history = betting_history

    def get_player(self, player_id: str) -> MockPlayer | None:
        return self.players_map.get(player_id)


# --- End Mock classes ---

# Use actual classes when available
GameState = MockGameState
Player = MockPlayer


RANK_TO_NUM = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
SUIT_TO_NUM = {"s": 1, "h": 2, "d": 3, "c": 4}  # Spades, Hearts, Diamonds, Clubs

ACTION_TO_ID = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4}
ROUND_TO_ID = {"pre-flop": 0, "flop": 1, "turn": 2, "river": 3}

# For feature construction as per prompt's examples for d_raw_feature=3:
# Card features: [encoded_card_value, 0, 0]
# Action features: [player_id_numeric, action_id_numeric, amount_normalized]
# Pot feature: [pot_value_normalized, type_id, 0] (type indicator 1)
# Current Bet feature: [bet_value_normalized, type_id, 0] (type indicator 2)
# Player Stack feature: [stack_value_normalized, type_id, 0] (type indicator 3)
# Round feature: [round_id, type_id, 0] (type indicator 4)

# Let's define these type indicators explicitly
TYPE_ID_CARD = 0.0  # Default type for cards, if needed in the third position.
TYPE_ID_POT = 1.0
TYPE_ID_CURRENT_BET = 2.0
TYPE_ID_PLAYER_STACK = 3.0
TYPE_ID_ROUND = 4.0
# Betting actions don't use a type_id in the third position in the prompt's
# example for d_raw_feature=3, as all three positions are used for
# player_id, action_id, amount.

class CardSetTransformer(nn.Module):
    """A lightweight Set Transformer for encoding unordered card sets.

    The module projects individual card features to ``hidden_dim`` and then
    applies a stack of self-attention layers.  The output is mean pooled to
    obtain a permutation invariant summary of the set.

    Parameters
    ----------
    card_dim:
        Dimension of the per-card feature vector (17 for one-hot rank/suit).
    hidden_dim:
        Internal hidden dimension used by the attention blocks.
    num_heads:
        Number of attention heads.
    num_layers:
        Number of Transformer encoder layers.
    """

    def __init__(
        self, card_dim: int, hidden_dim: int, num_heads: int = 1, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(card_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = hidden_dim

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """Encode a batch of card sets.

        Parameters
        ----------
        cards:
            Tensor of shape ``(batch, num_cards, card_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, hidden_dim)`` representing the pooled
            summary for each set.
        """

        if cards.numel() == 0:
            # No cards revealed yet. Return zero vector of appropriate dim.
            return cards.new_zeros((cards.size(0), self.output_dim))

        x = self.proj(cards)
        x = self.encoder(x)
        return x.mean(dim=1)


def _encode_card(card_str: str) -> list[float]:
    """Encode a card as one-hot rank and suit vectors.

    Returns a list of length 17: 13 for rank followed by 4 for suit.
    """
    if len(card_str) != 2:
        raise ValueError(f"Invalid card string: {card_str}")
    rank, suit = card_str[0].upper(), card_str[1].lower()
    ranks = list(RANK_TO_NUM.keys())
    suits = list(SUIT_TO_NUM.keys())
    if rank not in ranks or suit not in suits:
        raise ValueError(f"Invalid card components: {rank}, {suit}")
    rank_vec = [1.0 if r == rank else 0.0 for r in ranks]
    suit_vec = [1.0 if s == suit else 0.0 for s in suits]
    return rank_vec + suit_vec


def _get_numeric_player_id(
    player_id: str | int, all_player_ids_in_order: Sequence[str] | Mapping[str, int]
) -> int:
    """Convert ``player_id`` to its index in the ordered list.

    Parameters
    ----------
    player_id:
        Identifier of the player.  It may be provided as an ``int`` or ``str``
        depending on the game engine implementation.  We normalise it to a
        string for comparison.
    all_player_ids_in_order:
        Either a sequence of player identifiers (as strings) in seat order, or
        a mapping from player identifier string to its index.  Passing a
        mapping avoids repeated linear searches when this conversion is needed
        frequently.
    """

    pid_str = str(player_id)

    if isinstance(all_player_ids_in_order, Mapping):
        lookup = all_player_ids_in_order
    else:
        lookup = {str(pid): idx for idx, pid in enumerate(all_player_ids_in_order)}

    try:
        return lookup[pid_str]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(
            f"Player ID '{pid_str}' not found in the game's ordered player list."
        ) from exc


def _resolve_normalization_scale(
    game_state: GameState, explicit_scale: float | None
) -> float:
    """Return a positive chip scale for feature normalisation."""

    candidates: list[float | int | None] = [explicit_scale]
    rules = getattr(game_state, "rules", None)
    for source in (game_state, rules):
        if source is None:
            continue
        for attr in (
            "normalization_scale",
            "chip_normalization",
            "starting_stack",
            "big_blind",
            "small_blind",
        ):
            candidates.append(getattr(source, attr, None))

    for candidate in candidates:
        try:
            value = float(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value

    return 1.0


def infer_normalization_scale(
    game_state: GameState, explicit_scale: float | None = None
) -> float:
    """Infer a reasonable chip scale from ``game_state``.

    Parameters
    ----------
    game_state:
        The game instance or lightweight mock containing chip related
        configuration such as blinds or starting stacks.
    explicit_scale:
        Optional preferred scale (for example from configuration files).  When
        positive it takes precedence over values found on ``game_state``.
    """

    return _resolve_normalization_scale(game_state, explicit_scale)


def prepare_transformer_input(  # noqa: C901
    game_state: GameState,
    current_player_id: str | int,
    max_seq_len: int,
    d_raw_feature: int,
    normalization_scale: float | None = None,
    set_encoder: CardSetTransformer | None = None,
    return_mask: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Create tensors representing the infoset for ``current_player_id``.

    The returned tuple contains a summary of the player's hole cards, a
    summary of the community cards, and the sequential history features.  If
    ``return_mask`` is ``True`` an additional attention mask for the history
    sequence is provided.

    The ``normalization_scale`` argument defines the divisor used when
    normalising chip amounts such as bet sizes, pot size and stack depth.  If
    omitted the function inspects common configuration fields on ``game_state``
    (e.g. ``starting_stack`` or ``big_blind``) to pick a positive scale.
    """

    # ------------------------------------------------------------------
    # Extract relevant game attributes with fallbacks for the mock classes.
    # ------------------------------------------------------------------
    community_cards = getattr(game_state, "community_cards", None)
    if community_cards is None:
        community_cards = getattr(game_state.rules, "community_cards", [])

    pot = getattr(game_state, "pot", None)
    if pot is None:
        pot = getattr(game_state.rules, "pot", 0)

    current_bet = getattr(game_state, "current_bet", None)
    if current_bet is None:
        current_bet = getattr(game_state.rules, "current_bet", 0)

    betting_round = getattr(game_state, "betting_round", None)
    if betting_round is None or callable(betting_round):
        betting_round = getattr(game_state.rules, "betting_round", "pre-flop")

    betting_history = getattr(game_state, "betting_history", None)
    if betting_history is None:
        betting_history = getattr(game_state.rules, "betting_history", [])

    player_order = getattr(game_state, "player_order", None)
    if player_order is None:
        if hasattr(game_state, "players_map"):
            player_order = list(game_state.players_map.keys())
        else:
            num_players = getattr(game_state, "num_players", None)
            if num_players is None:
                num_players = getattr(getattr(game_state, "rules", None), "num_players", 0)
            player_order = list(range(int(num_players)))
    player_order = [str(pid) for pid in player_order]
    player_index_lookup = {pid: idx for idx, pid in enumerate(player_order)}

    original_player_id = current_player_id
    current_player_id = str(current_player_id)

    # ------------------------------------------------------------------
    # Encode card sets using the set encoder (or mean pooling fallback).
    # ------------------------------------------------------------------
    current_player_obj = None
    getter = getattr(game_state, "get_player", None)
    if callable(getter):
        for candidate in (original_player_id, current_player_id):
            try:
                current_player_obj = getter(candidate)
            except Exception:
                current_player_obj = None
            if current_player_obj is not None:
                break
    if current_player_obj is None:
        rules_view = getattr(game_state, "rules", game_state)
        try:
            idx = _get_numeric_player_id(current_player_id, player_index_lookup)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Player {current_player_id} not found in game_state.") from exc
        hands = getattr(rules_view, "hands", None)
        stacks = getattr(rules_view, "player_chips", None)
        bets = getattr(rules_view, "bets", None)
        if hands is None or stacks is None or bets is None:
            raise ValueError(f"Player {current_player_id} not found in game_state.")
        current_player_obj = Player(player_order[idx], hands[idx], int(stacks[idx]))
        setattr(current_player_obj, "current_bet_in_round", int(bets[idx]))

    hole_cards = torch.tensor(
        [_encode_card(c) for c in current_player_obj.hand], dtype=torch.float32
    )
    hole_cards = hole_cards.unsqueeze(0)  # batch dimension

    community_features = [_encode_card(c) for c in community_cards]
    if community_features:
        community_cards_tensor = torch.tensor(community_features, dtype=torch.float32)
    else:
        feature_dim = len(RANK_TO_NUM) + len(SUIT_TO_NUM)
        community_cards_tensor = torch.zeros((0, feature_dim), dtype=torch.float32)
    community_cards_tensor = community_cards_tensor.unsqueeze(0)

    if set_encoder is not None:
        hole_summary = set_encoder(hole_cards).squeeze(0)
        community_summary = set_encoder(community_cards_tensor).squeeze(0)
    else:  # simple mean pooling fallback
        hole_summary = hole_cards.mean(dim=1).squeeze(0)
        if community_cards_tensor.numel() > 0:
            community_summary = community_cards_tensor.mean(dim=1).squeeze(0)
        else:
            community_summary = torch.zeros_like(hole_summary)

    # ------------------------------------------------------------------
    # Encode betting history and scalar features as a sequence.
    # ------------------------------------------------------------------
    raw_sequence: list[list[float]] = []

    scale = _resolve_normalization_scale(game_state, normalization_scale)

    def _pad_feature(values: list[float]) -> list[float]:
        base = list(values)
        if len(base) >= d_raw_feature:
            return base[:d_raw_feature]
        return base + [0.0] * (d_raw_feature - len(base))

    # Betting history encoding
    for p_id_str, action_tuple in betting_history:
        action_name, amount_val = action_tuple
        numeric_p_id = float(_get_numeric_player_id(p_id_str, player_index_lookup))
        action_id = float(ACTION_TO_ID.get(action_name.lower(), -1))  # -1 for unknown
        normalized_amount = float(amount_val / scale if amount_val is not None else 0.0)
        raw_sequence.append(_pad_feature([numeric_p_id, action_id, normalized_amount]))

    # Pot size
    normalized_pot = float(pot / scale)
    raw_sequence.append(_pad_feature([normalized_pot, TYPE_ID_POT, 0.0]))

    # Current bet faced by player
    player_bet_in_round = getattr(current_player_obj, "current_bet_in_round", 0)
    effective_bet_faced = max(0, current_bet - player_bet_in_round)
    normalized_bet_faced = float(effective_bet_faced / scale)
    raw_sequence.append(
        _pad_feature([normalized_bet_faced, TYPE_ID_CURRENT_BET, 0.0])
    )

    # Player stack
    normalized_stack = float(current_player_obj.stack / scale)
    raw_sequence.append(
        _pad_feature([normalized_stack, TYPE_ID_PLAYER_STACK, 0.0])
    )

    # Betting round indicator
    round_id_numeric = float(ROUND_TO_ID.get(betting_round.lower(), -1))
    raw_sequence.append(_pad_feature([round_id_numeric, TYPE_ID_ROUND, 0.0]))

    # ------------------------------------------------------------------
    # Pad sequence and create mask
    # ------------------------------------------------------------------
    final_sequence: list[list[float]] = []
    attention_mask: list[int] = []
    for i in range(max_seq_len):
        if i < len(raw_sequence):
            final_sequence.append(raw_sequence[i])
            attention_mask.append(1)
        else:
            final_sequence.append([0.0] * d_raw_feature)
            attention_mask.append(0)

    history_tensor = torch.tensor(final_sequence, dtype=torch.float32)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.bool)

    if history_tensor.shape != (max_seq_len, d_raw_feature):  # pragma: no cover - sanity check
        raise ValueError(
            "Final tensor shape is "
            f"{history_tensor.shape}, expected ({max_seq_len}, {d_raw_feature})."
        )

    if return_mask:
        return hole_summary, community_summary, history_tensor, mask_tensor
    return hole_summary, community_summary, history_tensor
