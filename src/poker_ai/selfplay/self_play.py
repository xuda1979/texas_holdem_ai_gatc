"""
Implements External Sampling MCCFR for data generation as described in 'texas.tex'.
This version is simplified for Heads-Up No-Limit Hold'em (2 players).
"""
import torch
import copy
from typing import Dict, Any

# Assuming these imports are correct relative to the project structure
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.utils.state_representation import prepare_transformer_input
from poker_ai.utils.action_mapping import get_action_from_index, get_legal_actions_mask
from poker_ai.rules.cfr import calculate_strategy


class SelfPlay:
    """Orchestrates MCCFR traversals for training data generation."""

    def __init__(self, cfr_trainer, game_engine_config: Dict[str, Any]):
        self.cfr_trainer = cfr_trainer
        self.game_config = game_engine_config
        # Ensure this is a 2-player game for HU-NLHE as per the paper
        if self.game_config.get('num_players', 2) != 2:
            raise ValueError("This MCCFR implementation is designed for 2 players (HU-NLHE).")

    def play_hand_for_training(self, iteration: int):
        """
        Runs one full MCCFR traversal for a new hand, generating training data.
        """
        # 1. Initialize a new hand
        # Strategies are not used in MCCFR traversal, so they can be None
        game = TexasHoldem(
            num_players=self.game_config['num_players'],
            starting_stack=self.game_config['starting_stack']
        )
        game.initialize_game()

        # 2. Designate the traverser for this iteration
        # We alternate which player we are collecting training data for
        traverser_id = iteration % 2

        # 3. Start the recursive MCCFR traversal from the root of the game
        # The initial reach probabilities for both players are 1.0
        self._traverse_mccfr(game, traverser_id, iteration, p0=1.0, p1=1.0)

        # 4. After the traversal, run a training step on the collected data
        # The trainer's replay buffer is filled by the traversal.
        # We check if the buffer has enough samples to start training.
        if len(self.cfr_trainer.replay_buffer) > 256:
            loss = self.cfr_trainer.train(batch_size=256)
            if loss is not None:
                print(f"Iteration {iteration}: Training step complete. Loss: {loss:.4f}")

    def _get_policy(self, game: TexasHoldem, player_id: int) -> torch.Tensor:
        """
        Gets the current policy for a player at a given game state.
        This is done by querying the advantage network and applying regret matching.
        """
        # a. Get the infoset tensor for the current player
        # This function needs to be robust and handle the game state correctly
        model_config = self.cfr_trainer.config.get('model', {})
        max_seq_len = model_config.get('max_seq_len', 256)
        d_raw_feature = model_config.get('d_raw_feature', 18)
        state_tensor = prepare_transformer_input(game, player_id, max_seq_len, d_raw_feature)

        # b. Get advantages from the network
        advantages = self.cfr_trainer.get_advantages(state_tensor)

        # c. Get a mask for legal actions
        legal_actions_mask = get_legal_actions_mask(game, player_id, self.cfr_trainer.num_actions)

        # d. Apply the mask to the advantages by cloning and setting illegal entries to -inf.
        #    We avoid modifying the original tensor in-place to prevent unintended side-effects.
        masked_advantages = advantages.clone()
        masked_advantages[~legal_actions_mask] = float('-inf')

        # e. Convert advantages to a strategy via regret matching over legal actions only.
        #    We first compute the positive part of the masked advantages.  Negative or
        #    -inf values contribute zero to the sum.  If there is some positive regret
        #    among the legal actions, we normalize over those values.  Otherwise,
        #    we return a uniform distribution over the legal actions.  Illegal
        #    actions always receive zero probability.
        positive_adv = torch.clamp(masked_advantages, min=0.0)
        sum_positive = positive_adv.sum()
        policy = torch.zeros_like(masked_advantages)
        if sum_positive > 0:
            policy[legal_actions_mask] = positive_adv[legal_actions_mask] / sum_positive
        else:
            # If all advantages are non-positive, fall back to uniform distribution over legal actions.
            num_legal = int(legal_actions_mask.sum().item())
            if num_legal > 0:
                policy[legal_actions_mask] = 1.0 / num_legal
        return policy

    def _traverse_mccfr(self, game: TexasHoldem, traverser_id: int, iteration: int, p0: float, p1: float) -> float:
        """
        Recursive function to perform an External Sampling MCCFR traversal.
        - `p0`, `p1`: Reach probabilities for player 0 and 1 respectively.
        """
        # --- Terminal Node ---
        # Check if the hand is over (e.g., showdown, or one player folds)
        if game.is_hand_over():
            return game.get_payoff(traverser_id)

        # --- Chance Node ---
        # In this engine, chance events (dealing cards) are handled by advancing the stage
        if game.rules.betting_round_is_over():
            next_game = copy.deepcopy(game)
            if len(game.rules.community_cards) == 0:
                next_game.play_stage('flop')
            elif len(game.rules.community_cards) == 3:
                next_game.play_stage('turn')
            elif len(game.rules.community_cards) == 4:
                next_game.play_stage('river')
            return self._traverse_mccfr(next_game, traverser_id, iteration, p0, p1)

        # --- Decision Node ---
        current_player = game.rules.current_player
        policy = self._get_policy(game, current_player)

        if current_player == traverser_id:
            # --- Traverser's Node ---
            # We iterate over all legal actions to calculate regrets.
            node_value = 0.0
            action_utilities = torch.zeros(self.cfr_trainer.num_actions)

            legal_actions_mask = get_legal_actions_mask(game, current_player, self.cfr_trainer.num_actions)
            for action_idx in range(self.cfr_trainer.num_actions):
                if not legal_actions_mask[action_idx]:
                    continue

                # Create a new game state for this action
                next_game = copy.deepcopy(game)
                action_str, amount = get_action_from_index(action_idx, next_game, player_id=current_player)
                next_game.process_action(current_player, action_str, amount)
                next_game.rules.advance_turn()

                # The opponent's reach probability is not updated here
                reach_prob_p0, reach_prob_p1 = p0, p1

                # Recursively call to get the utility of this action
                action_utilities[action_idx] = self._traverse_mccfr(next_game, traverser_id, iteration, reach_prob_p0, reach_prob_p1)

            # Calculate node value using the current policy
            node_value = (action_utilities * policy).sum().item()

            # Calculate regrets and store in replay buffer
            regrets = action_utilities - node_value

            # Use the opponent's reach probability for weighting the regret update
            opponent_reach = p1 if traverser_id == 0 else p0
            weighted_regrets = regrets * opponent_reach

            model_config = self.cfr_trainer.config.get('model', {})
            max_seq_len = model_config.get('max_seq_len', 256)
            d_raw_feature = model_config.get('d_raw_feature', 18)
            state_tensor = prepare_transformer_input(game, traverser_id, max_seq_len, d_raw_feature)
            self.cfr_trainer.replay_buffer.push(state_tensor, weighted_regrets, iteration)

            return node_value
        else:
            # --- Opponent's Node ---
            # We sample one action and continue the traversal.
            action_idx = torch.multinomial(policy, 1).item()

            # Create the next game state
            next_game = copy.deepcopy(game)
            action_str, amount = get_action_from_index(action_idx, next_game, player_id=current_player)
            next_game.process_action(current_player, action_str, amount)
            next_game.rules.advance_turn()

            # Update reach probabilities for the sampled action
            if current_player == 0:
                return self._traverse_mccfr(next_game, traverser_id, iteration, p0 * policy[action_idx], p1)
            else:  # current_player == 1
                return self._traverse_mccfr(next_game, traverser_id, iteration, p0, p1 * policy[action_idx])