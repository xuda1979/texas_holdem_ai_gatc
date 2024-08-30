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

def compute_regrets(payoffs, action_values, actual_action):
    """
    Compute the regrets for each action.
    
    :param payoffs: The payoff values for the current state.
    :param action_values: The expected values of each action.
    :param actual_action: The action that was actually taken.
    :return: Regret values for each action.
    """
    regrets = payoffs - action_values
    return regrets

def regret_matching_plus(cumulative_regret, regrets, num_actions):
    """
    Perform the regret matching plus operation, which is an enhanced version of regret matching
    that ensures all regrets are non-negative.

    :param cumulative_regret: The cumulative regret tensor.
    :param regrets: The regret values for the current iteration.
    :param num_actions: The number of possible actions.
    :return: Updated strategy after applying regret matching plus.
    """
    cumulative_regret = update_regret(cumulative_regret, regrets)
    strategy = calculate_strategy(cumulative_regret, num_actions)
    return strategy, cumulative_regret

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
    for _ in range(num_iterations):
        current_strategy = calculate_strategy(cumulative_regret, num_actions)
        action_values = torch.zeros(num_actions)
        
        # Simulate action values based on game state and strategy
        for action in range(num_actions):
            action_values[action] = game.simulate_action(action)
        
        actual_action = game.get_actual_action()
        regrets = compute_regrets(action_values, action_values[actual_action], actual_action)
        current_strategy, cumulative_regret = regret_matching_plus(cumulative_regret, regrets, num_actions)
        cumulative_strategy = update_strategy(cumulative_strategy, current_strategy)

    return cumulative_regret, cumulative_strategy


