"""Cash-game table state management with table-stakes constraints."""

from __future__ import annotations

import random

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .rules import min_raise_to, split_winnings_with_odd_chips

_EPSILON = 1e-9


@dataclass
class CashPlayer:
    """State for a player seated in a cash game."""

    pid: int
    stack: float
    bankroll: float
    seated: bool = True
    in_hand: bool = False
    committed: float = 0.0
    folded: bool = False
    all_in: bool = False

    def available_to_bet(self) -> float:
        if not self.in_hand or self.folded:
            return 0.0
        return max(0.0, self.stack)

    def commit(self, amount: float) -> float:
        """Move chips from stack to the pot, enforcing table-stakes."""

        if amount <= 0 or not self.in_hand or self.folded:
            return 0.0
        move = min(amount, self.stack)
        if move <= 0:
            return 0.0
        self.stack -= move
        if self.stack <= _EPSILON:
            self.stack = 0.0
            self.all_in = True
        self.committed += move
        return move

    def reset_for_new_hand(self, *, active: bool) -> None:
        self.in_hand = active and self.seated and self.stack > _EPSILON
        self.committed = 0.0
        self.folded = False
        self.all_in = False


@dataclass
class PotLayer:
    """Represents a main or side pot capped at a contribution level."""

    cap_per_player: float
    contributors: List[int] = field(default_factory=list)
    eligible: List[int] = field(default_factory=list)
    amount: float = 0.0


@dataclass
class RaiseBounds:
    """Information about available raise sizes for an acting player."""

    min_total: float
    max_total: float
    min_reopens: bool
    max_reopens: bool


@dataclass
class LegalActions:
    """Summary of the legal options for a player on their turn."""

    can_fold: bool
    can_check: bool
    call_amount: Optional[float]
    raise_bounds: Optional[RaiseBounds]


@dataclass
class CashTable:
    """Implements casino-style table-stakes cash game rules."""

    small_blind: float
    big_blind: float
    min_buyin_bb: int = 40
    max_buyin_bb: int = 100
    min_bankroll_buyins: int = 1
    max_bankroll_buyins: int = 5
    rake_pct: float = 0.0
    rake_cap: float = 0.0
    no_flop_no_drop: bool = True
    players: Dict[int, CashPlayer] = field(default_factory=dict)
    button_idx: int = 0
    pot_layers: List[PotLayer] = field(default_factory=list)
    board_reached_flop: bool = False
    hand_start_stacks: Dict[int, float] = field(default_factory=dict)

    def seat_player(
        self,
        pid: int,
        bankroll: Optional[float] = None,
        buyin_bb: Optional[int] = None,
    ) -> None:
        if self.min_buyin_bb > self.max_buyin_bb:
            raise ValueError("min_buyin_bb cannot exceed max_buyin_bb")
        if self.min_bankroll_buyins > self.max_bankroll_buyins:
            raise ValueError("min_bankroll_buyins cannot exceed max_bankroll_buyins")

        if buyin_bb is None:
            buyin_bb = random.randint(self.min_buyin_bb, self.max_buyin_bb)
        if buyin_bb < self.min_buyin_bb or buyin_bb > self.max_buyin_bb:
            raise ValueError("buy-in outside allowed range")

        if bankroll is None:
            bankroll_buyins = random.randint(self.min_bankroll_buyins, self.max_bankroll_buyins)
            bankroll = bankroll_buyins * buyin_bb * self.big_blind
        if bankroll < 0:
            raise ValueError("bankroll must be non-negative")

        buyin_amount = min(bankroll, buyin_bb * self.big_blind)
        buyin_amount = max(0.0, buyin_amount)
        remaining = bankroll - buyin_amount
        if remaining < 0 and abs(remaining) <= _EPSILON:
            remaining = 0.0
        self.players[pid] = CashPlayer(
            pid=pid,
            stack=buyin_amount,
            bankroll=remaining,
        )

    def start_hand(self, active_pids: Iterable[int]) -> None:
        active = set(active_pids)
        for player in self.players.values():
            player.reset_for_new_hand(active=player.pid in active)
        self.pot_layers.clear()
        self.board_reached_flop = False
        self.hand_start_stacks = {
            pid: player.stack for pid, player in self.players.items() if player.in_hand
        }

    def stack_at_hand_start(self, pid: int) -> float:
        """Return the stack a player brought into the current hand."""

        return self.hand_start_stacks.get(pid, 0.0)

    def mark_flop_seen(self) -> None:
        self.board_reached_flop = True

    def post_blind(self, pid: int, amount: float) -> float:
        player = self.players.get(pid)
        if player is None:
            raise KeyError(pid)
        return player.commit(amount)

    def mark_fold(self, pid: int) -> None:
        player = self.players.get(pid)
        if player is None:
            raise KeyError(pid)
        if player.in_hand:
            player.folded = True

    def legal_actions(self, pid: int, to_call: float, last_raise: float) -> LegalActions:
        player = self.players.get(pid)
        if player is None:
            raise KeyError(pid)
        if not player.in_hand or player.folded:
            return LegalActions(can_fold=False, can_check=False, call_amount=None, raise_bounds=None)

        max_bet = player.available_to_bet()
        outstanding = max(0.0, to_call - player.committed)
        call_amount = min(outstanding, max_bet)
        can_check = outstanding <= _EPSILON
        raise_bounds: Optional[RaiseBounds] = None

        if max_bet - call_amount > _EPSILON:
            max_total = player.committed + max_bet
            try:
                min_raise_total = float(min_raise_to(int(round(to_call)), int(round(last_raise)), int(round(self.big_blind))))
            except ValueError:
                min_raise_total = to_call + max(last_raise, 0.0)
            min_raise_total = max(min_raise_total, player.committed + call_amount)
            min_raise_total = min(min_raise_total, max_total)
            if min_raise_total <= max_total + _EPSILON:
                min_reopens = (min_raise_total - to_call) >= max(last_raise, 0.0) - _EPSILON
                max_reopens = (max_total - to_call) >= max(last_raise, 0.0) - _EPSILON
                raise_bounds = RaiseBounds(
                    min_total=min_raise_total,
                    max_total=max_total,
                    min_reopens=min_reopens,
                    max_reopens=max_reopens,
                )

        call_value: Optional[float] = None
        if not can_check:
            call_value = call_amount

        return LegalActions(
            can_fold=True,
            can_check=can_check,
            call_amount=call_value,
            raise_bounds=raise_bounds,
        )

    def bet_or_raise_to(self, pid: int, target_total: float) -> float:
        player = self.players.get(pid)
        if player is None:
            raise KeyError(pid)
        if target_total < player.committed - _EPSILON:
            raise ValueError("target total must be >= current commitment")
        additional = target_total - player.committed
        return player.commit(additional)

    def _all_contributions(self) -> Dict[int, float]:
        contrib: Dict[int, float] = {}
        for pid, player in self.players.items():
            if player.in_hand and player.committed > _EPSILON:
                contrib[pid] = player.committed
        return contrib

    def build_side_pots(self) -> None:
        contributions = self._all_contributions()
        if not contributions:
            self.pot_layers.clear()
            return

        unique_levels = sorted({amount for amount in contributions.values() if amount > _EPSILON})
        if not unique_levels:
            self.pot_layers.clear()
            return

        self.pot_layers = []
        prev = 0.0
        for level in unique_levels:
            width = level - prev
            if width <= _EPSILON:
                prev = level
                continue
            contributors = [pid for pid, amt in contributions.items() if amt >= level - _EPSILON]
            eligible = [pid for pid in contributors if self.players[pid].in_hand and not self.players[pid].folded]
            if not contributors:
                prev = level
                continue
            amount = width * len(contributors)
            self.pot_layers.append(
                PotLayer(
                    cap_per_player=level,
                    contributors=contributors,
                    eligible=eligible,
                    amount=amount,
                )
            )
            prev = level

    def take_rake(self) -> float:
        total = sum(layer.amount for layer in self.pot_layers)
        if total <= _EPSILON:
            return 0.0
        if self.no_flop_no_drop and not self.board_reached_flop:
            return 0.0

        desired = total * self.rake_pct
        rake = min(desired, self.rake_cap)
        rake = round(rake, 2)
        if rake <= _EPSILON:
            return 0.0

        remaining = rake
        for layer in self.pot_layers:
            if remaining <= 0:
                break
            deduct = min(layer.amount, remaining)
            layer.amount -= deduct
            remaining -= deduct
        return rake - remaining

    def pay_out(self, winners_per_layer: Dict[int, Iterable[int]]) -> None:
        for idx, layer in enumerate(self.pot_layers):
            winners = list(winners_per_layer.get(idx, []))
            eligible = {pid for pid in layer.eligible}
            valid_winners = [pid for pid in winners if pid in eligible]
            if not valid_winners:
                continue
            split = split_winnings_with_odd_chips(int(round(layer.amount)), valid_winners, self.button_idx)
            for pid, amount in split.items():
                self.players[pid].stack += amount

    def top_up(self, pid: int, target_bb: int) -> float:
        player = self.players.get(pid)
        if player is None:
            raise KeyError(pid)
        target_amount = target_bb * self.big_blind
        missing = max(0.0, target_amount - player.stack)
        add = min(missing, player.bankroll)
        if add <= _EPSILON:
            return 0.0
        player.stack += add
        player.bankroll -= add
        return add

    @property
    def total_pot(self) -> float:
        return sum(layer.amount for layer in self.pot_layers)

