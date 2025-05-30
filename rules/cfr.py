import torch

def calculate_strategy(cumulative_regret, num_actions):
    """
    Calculate the strategy for the current iteration based on cumulative regret.
    
    :param cumulative_regret: Tensor representing the cumulative regret for each action.
    :param num_actions: The number of possible actions.
    :return: A probability distribution (strategy) over actions.
    """
    positive_regret = torch.clamp(cumulative_regret, min=0)
    sum_positive_regret = torch.sum(positive_regret)
    
    if sum_positive_regret > 0:
        return positive_regret / sum_positive_regret
    else:
        # If all regrets are non-positive, return a uniform random strategy
        return torch.ones(num_actions) / num_actions

def update_regret(cumulative_regret, regrets):
    """
    Update the cumulative regret values.
    
    :param cumulative_regret: The cumulative regret tensor to be updated.
    :param regrets: The regret values for the current iteration.
    :return: Updated cumulative regret tensor.
    """
    return cumulative_regret + regrets

def update_strategy(cumulative_strategy, current_strategy):
    """
    Update the cumulative strategy.
    
    :param cumulative_strategy: The cumulative strategy tensor to be updated.
    :param current_strategy: The strategy used in the current iteration.
    :return: Updated cumulative strategy tensor.
    """
    return cumulative_strategy + current_strategy

def compute_regrets(action_counterfactual_values, state_value):
    """
    Compute the regrets for each action.
    Regret for an action = (Counterfactual value of taking that action) - (Value of the current state/infoset).
    
    :param action_counterfactual_values: A tensor where element i is the counterfactual value
                                         of taking action i from the current state/infoset.
    :param state_value: The expected value of the current state/infoset, calculated based on
                        the current strategy.
    :return: Regret values for each action.
    """
    return action_counterfactual_values - state_value

# The 'regret_matching_plus' function has been removed as its logic was integrated
# directly into 'cfr_iteration' for clarity and correctness.
# This was done because the original implementation of cfr_iteration was overwriting
# the 'current_strategy' (used for regret calculation and accumulation) with the
# strategy intended for the *next* iteration, before accumulation.
# The corrected cfr_iteration now correctly:
# 1. Calculates current_strategy for *this* iteration using `calculate_strategy`.
# 2. Uses this current_strategy to derive action_counterfactual_values and the state_value.
# 3. Computes regrets for the current iteration.
# 4. Updates cumulative_regret using `update_regret`.
# 5. Accumulates the current_strategy (from step 1) using `update_strategy`.
# The strategy for the next iteration is then calculated at the beginning of the next loop iteration.

def compute_average_strategy(cumulative_strategy):
    """
    Compute the average strategy over all iterations.
    
    :param cumulative_strategy: The cumulative strategy tensor.
    :return: The average strategy tensor.
    """
    sum_strategy = torch.sum(cumulative_strategy)
    if sum_strategy > 0:
        return cumulative_strategy / sum_strategy
    else:
        # Return a uniform random strategy if no strategy has been accumulated
        num_actions = cumulative_strategy.size(0)
        return torch.ones(num_actions) / num_actions

def cfr_iteration(game, cumulative_regret, cumulative_strategy, num_actions, num_iterations):
    """
    Perform multiple iterations of the CFR algorithm.
    
    :param game: The game object representing the current state.
    :param cumulative_regret: The cumulative regret tensor.
    :param cumulative_strategy: The cumulative strategy tensor.
    :param num_actions: The number of possible actions.
    :param num_iterations: The number of CFR iterations to perform.
    :return: Updated cumulative regret and cumulative strategy tensors.
    """
    # This function processes a single information set (implicitly defined by the 'game' object's current state).
    # In a full CFR algorithm, you would typically traverse the game tree, and for each information set,
    # call a function similar to this or parts of its logic.

    for _ in range(num_iterations):
        # 1. Calculate current strategy based on cumulative regrets (Regret Matching)
        # current_strategy is a probability distribution over actions.
        current_strategy = calculate_strategy(cumulative_regret, num_actions)
        
        # 2. Calculate counterfactual values for each action and the value of the current state/infoset.
        # action_counterfactual_values[a] = value of taking action 'a' from the current infoset,
        # assuming all players play according to a fixed strategy profile (e.g., current strategy) thereafter.
        # game.simulate_action(action) is assumed to return this counterfactual value.
        action_counterfactual_values = torch.zeros(num_actions)
        for action_idx in range(num_actions):
            # It's crucial that game.simulate_action(action_idx) returns the counterfactual payoff
            # for taking action_idx from the current game state, playing out to a terminal node.
            # The game state should be consistent for each call within this loop for the same infoset.
            action_counterfactual_values[action_idx] = game.simulate_action(action_idx)

        # The value of the current state/infoset is the expected value of its action counterfactual values,
        # weighted by the probability of taking each action under the current strategy.
        # state_value = sum(current_strategy[a] * action_counterfactual_values[a] for a in actions)
        state_value = torch.sum(current_strategy * action_counterfactual_values)

        # 3. Compute regrets for each action.
        # Regret for an action 'a' = action_counterfactual_values[a] - state_value.
        # The game.get_actual_action() call was removed as it's not used in standard immediate regret calculation
        # for all actions in an information set. If sampling is used (e.g. Monte Carlo CFR),
        # then only one action's regret might be updated, but this function seems to update all.
        regrets = compute_regrets(action_counterfactual_values, state_value)

        # 4. Update cumulative regrets and strategy for the next iteration (using Regret Matching Plus variant)
        # The 'regret_matching_plus' function updates cumulative_regret and recalculates strategy.
        # The returned 'current_strategy' from regret_matching_plus is the one for the *next* step,
        # but we need to accumulate the one used for *this* iteration's calculations.
        # So, we pass the 'regrets' to update 'cumulative_regret', and then
        # 'calculate_strategy' inside 'regret_matching_plus' will give the strategy for the next round.
        # The strategy to be accumulated for averaging is the 'current_strategy' calculated at step 1.

        # Update cumulative regrets
        cumulative_regret = update_regret(cumulative_regret, regrets) # cumulative_regret += regrets
        
        # Accumulate the strategy used in this iteration for averaging later
        # Note: In some CFR variants (like CFR+), the strategy used for accumulation
        # is weighted by the iteration number or other factors. Here, it's a direct sum.
        cumulative_strategy = update_strategy(cumulative_strategy, current_strategy)

    return cumulative_regret, cumulative_strategy


