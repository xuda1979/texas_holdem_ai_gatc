"""Head-to-head match evaluation for full Texas Hold'em strategies.

Exact best response is intractable on full Hold'em, so training progress is
tracked by playing matches against fixed baseline opponents and reporting the
win rate in big blinds per 100 hands (bb/100), the standard poker metric.

Variance reduction uses duplicate dealing: every deck seed is played twice
with the seats swapped, so card luck largely cancels and the residual signal
is skill.  For deterministic strategies a self-match scores exactly 0 bb/100.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from poker_ai.engine.texas_holdem import TexasHoldem


class Strategy(Protocol):
    """Anything implementing the existing ``PlayerStrategy`` protocol."""

    def choose_action(self, game: TexasHoldem, player_index: int) -> tuple[str, int | None]:
        ...


@dataclass
class MatchResult:
    """Outcome of a duplicate head-to-head match, from player A's perspective."""

    hands_played: int
    total_chips: float
    big_blind: float
    bb_per_100: float
    stderr_bb_per_100: float

    def is_significantly_positive(self, z: float = 1.96) -> bool:
        return self.bb_per_100 - z * self.stderr_bb_per_100 > 0.0

    def is_significantly_negative(self, z: float = 1.96) -> bool:
        return self.bb_per_100 + z * self.stderr_bb_per_100 < 0.0


def _is_betting_round_over(game: TexasHoldem) -> bool:
    rules = game.rules
    active_non_allin = [
        i
        for i in range(rules.num_players)
        if rules.active_players[i] and rules.player_chips[i] > 0
    ]
    if not active_non_allin:
        return True
    all_settled = all(rules.bets[i] == rules.current_bet for i in active_non_allin)
    actions_this_round = getattr(rules, "actions_this_round", 0)
    return all_settled and actions_this_round >= len(active_non_allin)


def _advance_street(game: TexasHoldem) -> None:
    cleanup = getattr(game.rules, "end_betting_round_cleanup", None)
    if callable(cleanup):
        cleanup()

    community = game.rules.community_cards
    if len(community) == 0:
        game.play_stage("flop")
    elif len(community) == 3:
        game.play_stage("turn")
    elif len(community) == 4:
        game.play_stage("river")

    start = (game.rules.dealer_button + 1) % game.rules.num_players
    current = start
    for _ in range(game.rules.num_players):
        if game.rules.active_players[current] and game.rules.player_chips[current] > 0:
            break
        current = (current + 1) % game.rules.num_players
    game.rules.current_player = current


def play_hand(
    strategies: list[Strategy],
    *,
    starting_stack: int = 1000,
    big_blind: int = 10,
    small_blind: int = 5,
    deck_seed: int | None = None,
    max_actions: int = 500,
) -> list[float]:
    """Play one hand and return each player's chip payoff."""

    num_players = len(strategies)
    game = TexasHoldem(num_players=num_players, starting_stack=starting_stack, verbose=False)
    game.rules.big_blind = big_blind
    game.rules.small_blind = small_blind
    if deck_seed is not None:
        game.rules.deck_manager.rng.seed(deck_seed)
    game.initialize_game()

    actions_taken = 0
    while not game.is_hand_over():
        if _is_betting_round_over(game):
            _advance_street(game)
            continue
        actions_taken += 1
        if actions_taken > max_actions:  # engine stuck; abort defensively
            raise RuntimeError("Hand exceeded maximum action count")
        player = game.rules.current_player
        action_str, amount = strategies[player].choose_action(game, player)
        game.process_action(player, action_str, amount)
        game.rules.advance_turn()

    return [float(game.get_payoff(pid)) for pid in range(num_players)]


def play_match(
    strategy_a: Strategy,
    strategy_b: Strategy,
    num_hands: int = 200,
    *,
    starting_stack: int = 1000,
    big_blind: int = 10,
    small_blind: int = 5,
    seed: int = 0,
) -> MatchResult:
    """Play a duplicate heads-up match and return A's bb/100 with a std error.

    ``num_hands`` is rounded up to an even number; each deck seed is played
    once with A in seat 0 and once with A in seat 1.
    """

    rng = random.Random(seed)
    pairs = (num_hands + 1) // 2
    pair_scores: list[float] = []  # A's average chips per hand within a pair
    total_chips = 0.0

    for _ in range(pairs):
        deck_seed = rng.randrange(1 << 63)
        payoff_a_seat0 = play_hand(
            [strategy_a, strategy_b],
            starting_stack=starting_stack,
            big_blind=big_blind,
            small_blind=small_blind,
            deck_seed=deck_seed,
        )[0]
        payoff_a_seat1 = play_hand(
            [strategy_b, strategy_a],
            starting_stack=starting_stack,
            big_blind=big_blind,
            small_blind=small_blind,
            deck_seed=deck_seed,
        )[1]
        pair_total = payoff_a_seat0 + payoff_a_seat1
        total_chips += pair_total
        pair_scores.append(pair_total / 2.0)

    hands_played = pairs * 2
    mean_chips_per_hand = total_chips / hands_played
    bb_per_100 = mean_chips_per_hand / big_blind * 100.0

    if len(pair_scores) > 1:
        mean_pair = sum(pair_scores) / len(pair_scores)
        var = sum((s - mean_pair) ** 2 for s in pair_scores) / (len(pair_scores) - 1)
        stderr_chips = math.sqrt(var / len(pair_scores))
    else:
        stderr_chips = float("inf")
    stderr_bb_per_100 = stderr_chips / big_blind * 100.0

    return MatchResult(
        hands_played=hands_played,
        total_chips=total_chips,
        big_blind=float(big_blind),
        bb_per_100=bb_per_100,
        stderr_bb_per_100=stderr_bb_per_100,
    )


class AlwaysFoldStrategy:
    """Folds whenever facing a bet; otherwise checks.  The weakest baseline."""

    @property
    def is_human(self) -> bool:
        return False

    def choose_action(self, game: TexasHoldem, player_index: int) -> tuple[str, int | None]:
        actions = game.get_valid_actions(player_index)
        if "check" in actions:
            return "check", None
        return "fold", None


class CallingStationStrategy:
    """Checks or calls every decision; never bets, raises or folds."""

    @property
    def is_human(self) -> bool:
        return False

    def choose_action(self, game: TexasHoldem, player_index: int) -> tuple[str, int | None]:
        actions = game.get_valid_actions(player_index)
        if "check" in actions:
            return "check", None
        if "call" in actions:
            return "call", None
        return "fold", None
