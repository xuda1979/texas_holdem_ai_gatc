"""Implements External Sampling MCCFR for data generation as described in ``texas.tex``."""

# ruff: noqa

import copy
import random
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

# Assuming these imports are correct relative to the project structure
from poker_ai.engine.texas_holdem import TexasHoldem, TexasHoldemRules
from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import (
    infer_normalization_scale,
    prepare_transformer_input,
)


@dataclass
class _GameStateSnapshot:
    """Lightweight snapshot for restoring ``TexasHoldem`` traversal state."""

    rules: TexasHoldemRules
    end_game_early: bool
    winner: Any
    pending_showdown: Any


@dataclass
class _GameStateSnapshot:
    """Lightweight snapshot for restoring ``TexasHoldem`` traversal state."""

    rules: TexasHoldemRules
    end_game_early: bool
    winner: Any
    pending_showdown: Any


class SelfPlay:
    """Orchestrates MCCFR traversals for training data generation."""

    def __init__(
        self,
        cfr_trainer: object,
        game_engine_config: dict[str, Any],
        training_config: dict[str, Any] | None = None,
    ) -> None:
        self.cfr_trainer = cfr_trainer
        self.starting_stack = game_engine_config.get("starting_stack", 1000)
        self.big_blind = game_engine_config.get("big_blind", 10)
        self.small_blind = game_engine_config.get("small_blind", 5)
        self.min_players = game_engine_config.get("min_players", 2)
        self.max_players = game_engine_config.get("max_players", 10)
        cfg = training_config or {}
        min_buffer_raw = cfg.get("min_buffer_before_train", 256)
        try:
            min_buffer = int(min_buffer_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            min_buffer = 256
        self.min_buffer_before_train = max(1, min_buffer)
 

    def _normalization_scale_for_game(self, game: TexasHoldem) -> float:
        """Determine the chip normalization scale for ``game``."""

        config_obj = getattr(self.cfr_trainer, "config", {})
        preferred_scale = None
        if isinstance(config_obj, dict):
            for key in ("normalization_scale", "chip_normalization", "starting_stack"):
                value = config_obj.get(key)
                if isinstance(value, (int, float)):
                    preferred_scale = float(value)
                    break

        if preferred_scale is None:
            preferred_scale = float(self.starting_stack)

        return infer_normalization_scale(game, preferred_scale)
 

    def play_hand_for_training(self, iteration: int = 0) -> list[Any]:
        """Run one full MCCFR traversal for a new hand."""
        # 1. Initialize a new hand with a random number of players
        num_players = random.randint(self.min_players, self.max_players)
        game = TexasHoldem(num_players=num_players, starting_stack=self.starting_stack)
        game.rules.big_blind = self.big_blind
        game.rules.small_blind = self.small_blind
        game.initialize_game()

        # 2. Perform a traversal for each player in the hand
        base_reach = [1.0] * num_players
        for traverser_id in range(num_players):
            snapshot = self._snapshot_state(game)
            self._traverse_mccfr(game, traverser_id, iteration, base_reach.copy(), [])
            self._restore_state(game, snapshot)

        # 3. After the traversals, run a training step on the collected data

        if len(self.cfr_trainer.replay_buffer) >= self.min_buffer_before_train:
            loss = self.cfr_trainer.train(batch_size=self.min_buffer_before_train)
            if loss is not None:
                print(f"Iteration {iteration}: Training step complete. Loss: {loss:.4f}")

        return self.cfr_trainer.replay_buffer

    def _get_policy(self, game: TexasHoldem, player_id: int) -> torch.Tensor:
        """
        Gets the current policy for a player at a given game state.
        This is done by querying the advantage network and applying regret matching.
        """
        # a. Get the infoset tensor for the current player
        # This function needs to be robust and handle the game state correctly
        model_config = self.cfr_trainer.config.get("model", {})
        max_seq_len = model_config.get("max_seq_len", 256)
        d_raw_feature = model_config.get("d_raw_feature", 18)
        normalization_scale = self._normalization_scale_for_game(game)
        hole, community, history_tensor = prepare_transformer_input(
            game,
            player_id,
            max_seq_len,
            d_raw_feature,
            normalization_scale=normalization_scale,
        )

        # b. Get advantages from the network (support older trainer signatures)
        try:
            advantages = self.cfr_trainer.get_advantages(hole, community, history_tensor)
        except TypeError:  # pragma: no cover - backwards compat
            advantages = self.cfr_trainer.get_advantages(history_tensor)

        # c. Get a mask for legal actions
        legal_actions_mask = get_legal_actions_mask(
            game, player_id, self.cfr_trainer.num_actions
        )
        legal_actions_mask = legal_actions_mask.to(advantages.device)

        # d. Regret matching over legal actions only while avoiding propagating -inf/NaN
        #    when masking out illegal moves.
        positive_advantages = torch.clamp(advantages, min=0.0)
        positive_advantages = torch.where(
            legal_actions_mask,
            positive_advantages,
            torch.zeros_like(positive_advantages, device=advantages.device),
        )

        # e. Normalize across legal actions.  If all regrets are non-positive, revert to
        #    a uniform policy over legal actions only.
        sum_positive = positive_advantages.sum()
        policy = torch.zeros_like(advantages, device=advantages.device)
        if sum_positive.item() > 0:
            policy[legal_actions_mask] = positive_advantages[legal_actions_mask] / sum_positive
        else:
            num_legal = int(legal_actions_mask.sum().item())
            if num_legal > 0:
                policy[legal_actions_mask] = 1.0 / num_legal
        return policy

    def _is_betting_round_over(self, game: TexasHoldem) -> bool:
        """Return ``True`` when the current street has finished for traversal purposes."""

        rules = game.rules
        active_non_allin = [
            i
            for i in range(rules.num_players)
            if rules.active_players[i] and rules.player_chips[i] > 0
        ]
        if not active_non_allin:
            return True

        all_settled = all(
            rules.bets[i] == rules.current_bet for i in active_non_allin
        )
        actions_this_round = getattr(rules, "actions_this_round", 0)
        return all_settled and actions_this_round >= len(active_non_allin)

    def _advance_street(self, game: TexasHoldem) -> None:
        """Advance the existing game to the next street without allocating a clone."""

        game.rules.end_betting_round_cleanup()

        community_cards = game.rules.community_cards
        if len(community_cards) == 0:
            game.play_stage("flop")
        elif len(community_cards) == 3:
            game.play_stage("turn")
        elif len(community_cards) == 4:
            game.play_stage("river")

        # First player to act post-flop is directly left of the dealer button.
        start_player = (game.rules.dealer_button + 1) % game.rules.num_players
        current = start_player
        for _ in range(game.rules.num_players):
            if game.rules.active_players[current] and game.rules.player_chips[current] > 0:
                break
            current = (current + 1) % game.rules.num_players
        game.rules.current_player = current

    def _snapshot_state(self, game: TexasHoldem) -> _GameStateSnapshot:
        """Capture the current mutable state so it can be restored later."""

        return _GameStateSnapshot(
            rules=game.rules.clone(),
            end_game_early=game.end_game_early,
            winner=copy.deepcopy(game.winner),
            pending_showdown=copy.deepcopy(getattr(game, "_pending_showdown_winnings", None)),
        )

    def _restore_state(self, game: TexasHoldem, snapshot: _GameStateSnapshot) -> None:
        """Restore ``game`` to a previously captured snapshot."""

        game.rules = snapshot.rules
        game.end_game_early = snapshot.end_game_early
        game.winner = snapshot.winner
        if hasattr(game, "_pending_showdown_winnings"):
            game._pending_showdown_winnings = snapshot.pending_showdown

    def _apply_action_in_place(
        self, game: TexasHoldem, player_id: int, action_idx: int
    ) -> None:
        """Apply an indexed action to ``game`` without cloning."""

        action = get_action_from_index(action_idx, game, player_id=player_id)
        action_str, amount = action_to_tuple(action)
        game.process_action(player_id, action_str, amount)
        game.rules.advance_turn()

    def _traverse_mccfr(  # noqa: C901
        self,
        game: TexasHoldem,
        traverser_id: int,
        iteration: int,
        reach_probs: list[float],
        undo_stack: Optional[List[_GameStateSnapshot]] = None,
    ) -> float:
        """Recursive function to perform an External Sampling MCCFR traversal."""

        if undo_stack is None:
            undo_stack = []

        # --- Terminal Node ---
        if game.is_hand_over():
            return game.get_payoff(traverser_id)

        # --- Chance Node ---
        if self._is_betting_round_over(game):
            undo_stack.append(self._snapshot_state(game))
            self._advance_street(game)
            value = self._traverse_mccfr(
                game, traverser_id, iteration, reach_probs, undo_stack
            )
            snapshot = undo_stack.pop()
            self._restore_state(game, snapshot)
            return value

        # --- Decision Node ---
        current_player = game.rules.current_player
        policy = self._get_policy(game, current_player)

        if current_player == traverser_id:
            action_utilities = torch.zeros(self.cfr_trainer.num_actions)

            legal_actions_mask = get_legal_actions_mask(
                game, current_player, self.cfr_trainer.num_actions
            )
            for action_idx in range(self.cfr_trainer.num_actions):
                if not legal_actions_mask[action_idx]:
                    continue

                undo_stack.append(self._snapshot_state(game))
                self._apply_action_in_place(game, current_player, action_idx)
                action_utilities[action_idx] = self._traverse_mccfr(
                    game, traverser_id, iteration, reach_probs, undo_stack
                )
                snapshot = undo_stack.pop()
                self._restore_state(game, snapshot)

            node_value = (action_utilities * policy).sum().item()
            regrets = action_utilities - node_value

            opponent_reach = 1.0
            for idx, prob in enumerate(reach_probs):
                if idx != traverser_id:
                    opponent_reach *= prob
            weighted_regrets = regrets * opponent_reach

            model_config = self.cfr_trainer.config.get("model", {})
            max_seq_len = model_config.get("max_seq_len", 256)
            d_raw_feature = model_config.get("d_raw_feature", 18)
            hole_s, community_s, state_tensor = prepare_transformer_input(
                game, traverser_id, max_seq_len, d_raw_feature
            )
            if hasattr(self.cfr_trainer, "replay_buffer"):
                try:
                    self.cfr_trainer.replay_buffer.push(
                        hole_s,
                        community_s,
                        state_tensor,
                        weighted_regrets,
                        action_utilities.detach().clone(),
                        legal_actions_mask,
                        iteration,
                    )
                except TypeError:
                    self.cfr_trainer.replay_buffer.push(
                        hole_s, community_s, state_tensor, weighted_regrets, iteration
                    )

            return node_value

        action_idx = torch.multinomial(policy, 1).item()
        undo_stack.append(self._snapshot_state(game))
        self._apply_action_in_place(game, current_player, action_idx)
        new_reach = reach_probs.copy()
        new_reach[current_player] *= policy[action_idx].item()
        value = self._traverse_mccfr(game, traverser_id, iteration, new_reach, undo_stack)
        snapshot = undo_stack.pop()
        self._restore_state(game, snapshot)
        return value
