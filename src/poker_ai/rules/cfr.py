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


def compute_regrets(action_values, state_value):
    """Compute regret for each action.

    Counterfactual regret for an action is the difference between the value of
    taking that action and the state's value under the current strategy.  The
    previous implementation accepted an ``actual_action`` argument and
    subtracted the value of that action from every other action, which does not
    match the definition used in CFR.

    Parameters
    ----------
    action_values : torch.Tensor
        Expected value of each available action.
    state_value : torch.Tensor | float
        The value of the state under the mixed strategy.  Typically this is a
        scalar computed as ``(strategy * action_values).sum()``.

    Returns
    -------
    torch.Tensor
        Regret for every action.
    """
    return action_values - state_value


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


def cfr_plus_iteration(
    game, cumulative_regret, cumulative_strategy, num_actions, num_iterations, prune_threshold=0.0
):
    """Run CFR+ iterations with optional pruning of low-regret actions."""
    for _ in range(num_iterations):
        current_strategy = calculate_strategy(cumulative_regret, num_actions)
        action_values = torch.zeros(num_actions)

        for action in range(num_actions):
            action_values[action] = game.simulate_action(action)

        state_value = torch.sum(current_strategy * action_values)
        regrets = compute_regrets(action_values, state_value)

        if prune_threshold > 0.0:
            mask = cumulative_regret < prune_threshold
            regrets = torch.where(mask, torch.zeros_like(regrets), regrets)

        current_strategy, cumulative_regret = regret_matching_plus(
            cumulative_regret, regrets, num_actions
        )
        cumulative_strategy = update_strategy(cumulative_strategy, current_strategy)

        cumulative_regret = torch.clamp(cumulative_regret, min=0)

    return cumulative_regret, cumulative_strategy


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

        state_value = torch.sum(current_strategy * action_values)
        regrets = compute_regrets(action_values, state_value)
        current_strategy, cumulative_regret = regret_matching_plus(
            cumulative_regret, regrets, num_actions
        )
        cumulative_strategy = update_strategy(cumulative_strategy, current_strategy)

    return cumulative_regret, cumulative_strategy


def discounted_cfr_plus_iteration(
    game, cumulative_regret, cumulative_strategy, num_actions, num_iterations, discount=1.0
):
    """CFR+ iteration with a discount factor applied to historical regrets."""
    for _ in range(num_iterations):
        current_strategy = calculate_strategy(cumulative_regret, num_actions)
        action_values = torch.zeros(num_actions)
        for action in range(num_actions):
            action_values[action] = game.simulate_action(action)
        state_value = torch.sum(current_strategy * action_values)
        regrets = compute_regrets(action_values, state_value)
        cumulative_regret.mul_(discount)
        current_strategy, cumulative_regret = regret_matching_plus(
            cumulative_regret, regrets, num_actions
        )
        cumulative_strategy = update_strategy(cumulative_strategy, current_strategy)
        cumulative_regret = torch.clamp(cumulative_regret, min=0)
    return cumulative_regret, cumulative_strategy
