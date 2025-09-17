"""Implements External Sampling MCCFR for data generation as described in ``texas.tex``."""

# ruff: noqa

import random
from typing import Any

import torch

# Assuming these imports are correct relative to the project structure
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import prepare_transformer_input


class SelfPlay:
    """Orchestrates MCCFR traversals for training data generation."""

    def __init__(self, cfr_trainer: object, game_engine_config: dict[str, Any]) -> None:
        self.cfr_trainer = cfr_trainer
        self.starting_stack = game_engine_config.get("starting_stack", 1000)
        self.big_blind = game_engine_config.get("big_blind", 10)
        self.small_blind = game_engine_config.get("small_blind", 5)
        self.min_players = game_engine_config.get("min_players", 2)
        self.max_players = game_engine_config.get("max_players", 10)

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
            self._traverse_mccfr(game.clone(), traverser_id, iteration, base_reach.copy())

        # 3. After the traversals, run a training step on the collected data

        if len(self.cfr_trainer.replay_buffer) >= 256:
            loss = self.cfr_trainer.train(batch_size=256)
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
        normalization_scale = getattr(
            game.rules,
            "starting_stack",
            getattr(game, "starting_stack", self.starting_stack),
        )
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

    def _advance_street(self, game: TexasHoldem) -> TexasHoldem:
        """Return a cloned game state advanced to the next street with clean betting state."""

        next_game = game.clone()
        next_game.rules.end_betting_round_cleanup()

        community_cards = next_game.rules.community_cards
        if len(community_cards) == 0:
            next_game.play_stage("flop")
        elif len(community_cards) == 3:
            next_game.play_stage("turn")
        elif len(community_cards) == 4:
            next_game.play_stage("river")

        # First player to act post-flop is directly left of the dealer button.
        start_player = (next_game.rules.dealer_button + 1) % next_game.rules.num_players
        current = start_player
        for _ in range(next_game.rules.num_players):
            if (
                next_game.rules.active_players[current]
                and next_game.rules.player_chips[current] > 0
            ):
                break
            current = (current + 1) % next_game.rules.num_players
        next_game.rules.current_player = current
        return next_game

    def _traverse_mccfr(  # noqa: C901
        self,
        game: TexasHoldem,
        traverser_id: int,
        iteration: int,
        reach_probs: list[float],
    ) -> float:
        """Recursive function to perform an External Sampling MCCFR traversal."""
        # --- Terminal Node ---
        # Check if the hand is over (e.g., showdown, or one player folds)
        if game.is_hand_over():
            return game.get_payoff(traverser_id)

        # --- Chance Node ---
        if self._is_betting_round_over(game):
            next_game = self._advance_street(game)
            return self._traverse_mccfr(next_game, traverser_id, iteration, reach_probs)

        # --- Decision Node ---
        current_player = game.rules.current_player
        policy = self._get_policy(game, current_player)

        if current_player == traverser_id:
            # --- Traverser's Node ---
            # We iterate over all legal actions to calculate regrets.
            node_value = 0.0
            action_utilities = torch.zeros(self.cfr_trainer.num_actions)

            legal_actions_mask = get_legal_actions_mask(
                game, current_player, self.cfr_trainer.num_actions
            )
            for action_idx in range(self.cfr_trainer.num_actions):
                if not legal_actions_mask[action_idx]:
                    continue

                # Create a new game state for this action
                next_game = game.clone()
                action = get_action_from_index(action_idx, next_game, player_id=current_player)
                action_str, amount = action_to_tuple(action)
                next_game.process_action(current_player, action_str, amount)
                next_game.rules.advance_turn()

                # Recursively call to get the utility of this action
                action_utilities[action_idx] = self._traverse_mccfr(
                    next_game, traverser_id, iteration, reach_probs.copy()
                )

            # Calculate node value using the current policy
            node_value = (action_utilities * policy).sum().item()

            # Calculate regrets and store in replay buffer
            regrets = action_utilities - node_value

            # Use the opponent's reach probability for weighting the regret update
            opponent_reach = 1.0
            for idx, prob in enumerate(reach_probs):
                if idx != traverser_id:
                    opponent_reach *= prob
            weighted_regrets = regrets * opponent_reach

            model_config = self.cfr_trainer.config.get("model", {})
            max_seq_len = model_config.get("max_seq_len", 256)
            d_raw_feature = model_config.get("d_raw_feature", 18)
            normalization_scale = getattr(
                game.rules,
                "starting_stack",
                getattr(game, "starting_stack", self.starting_stack),
            )
            hole_s, community_s, state_tensor = prepare_transformer_input(
                game,
                traverser_id,
                max_seq_len,
                d_raw_feature,
                normalization_scale=normalization_scale,
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
                    # Older replay buffers accept only regret targets.
                    self.cfr_trainer.replay_buffer.push(
                        hole_s, community_s, state_tensor, weighted_regrets, iteration
                    )

            return node_value
        else:
            # --- Opponent's Node ---
            # We sample one action and continue the traversal.
            action_idx = torch.multinomial(policy, 1).item()

            # Create the next game state
            next_game = game.clone()
            action = get_action_from_index(action_idx, next_game, player_id=current_player)
            action_str, amount = action_to_tuple(action)
            next_game.process_action(current_player, action_str, amount)
            next_game.rules.advance_turn()

            # Update reach probabilities for the sampled action
            new_reach = reach_probs.copy()
            new_reach[current_player] *= policy[action_idx].item()
            return self._traverse_mccfr(next_game, traverser_id, iteration, new_reach)
