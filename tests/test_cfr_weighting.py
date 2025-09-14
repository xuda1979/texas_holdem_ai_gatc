import numpy as np
from poker_ai.ai.trainers.cfr_core import (
    regret_matching_plus,
    update_regret,
    update_strategy_sum,
    average_strategy,
)


def test_regret_matching_plus_uniform_when_all_nonpositive():
    r = np.array([-1.0, 0.0, -2.0])
    sigma = regret_matching_plus(r)
    assert np.allclose(sigma, np.array([1/3, 1/3, 1/3]))


def test_regret_and_strategy_updates_shapes_and_weights():
    # Toy node with 3 actions
    cum_regrets = np.zeros(3)
    strat_sum = np.zeros(3)

    # Suppose current policy and resulting utilities:
    sigma = np.array([0.2, 0.3, 0.5])
    action_utils = np.array([1.0, 0.5, -0.5])
    node_util = float((sigma * action_utils).sum())

    # One CFR iteration at this infoset:
    # Opponent reach for regret; player reach for strategy averaging.
    opp_reach = 0.8
    player_reach = 0.6
    iter_w = 5.0  # CFR+ style iteration weighting

    update_regret(cum_regrets, action_utils, node_util, opp_reach, cfr_plus=True)
    update_strategy_sum(strat_sum, sigma, player_reach, iteration_weight=iter_w)

    # Regrets should be non-negative (CFR+ clipping) and proportional to opp_reach
    assert np.all(cum_regrets >= -1e-12)
    # Strategy sums should scale with player reach and iteration weight
    assert np.allclose(strat_sum, iter_w * player_reach * sigma)

    # Average strategy must normalize
    avg = average_strategy(strat_sum)
    assert np.isclose(avg.sum(), 1.0)
