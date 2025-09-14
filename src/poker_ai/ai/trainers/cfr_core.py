from __future__ import annotations
from typing import Iterable
import numpy as np

"""
Minimal CFR / CFR+ building blocks with correct reach-weighting.
Use these from your tabular CFR or Deep-CFR trainers.

References:
  - Zinkevich et al., NIPS 2007 (CFR).
  - Tammelin, CFR+ (2014).
  - Brown et al., ICML 2019 (Deep CFR).
"""


def regret_matching_plus(regrets: np.ndarray) -> np.ndarray:
    """
    CFR+ policy from cumulative regrets.
    If all regrets <= 0, returns uniform.
    regrets: shape [A], may contain negatives (they are treated as 0 here).
    """
    pos = np.maximum(regrets, 0.0)
    s = float(pos.sum())
    if s <= 0.0:
        return np.full_like(pos, 1.0 / pos.size)
    return pos / s


def update_regret(
    cum_regrets: np.ndarray,
    action_utils: np.ndarray,
    node_util: float,
    opponent_reach: float,
    cfr_plus: bool = True,
) -> None:
    """
    cum_regrets += π_{-i} * (Q(a) - V)
      - action_utils: shape [A], counterfactual values V_i(I->a)
      - node_util: scalar V_i(I) = sum_a σ(a) * V_i(I->a)
      - opponent_reach: π_{-i}(h[I])
    CFR+ keeps regrets non-negative after *cumulative* update.
    """
    delta = opponent_reach * (action_utils - node_util)
    cum_regrets += delta
    if cfr_plus:
        np.maximum(cum_regrets, 0.0, out=cum_regrets)


def update_strategy_sum(
    strat_sum: np.ndarray,
    strategy: np.ndarray,
    player_reach: float,
    iteration_weight: float = 1.0,
) -> None:
    """
    S += w_t * π_i * σ
    """
    strat_sum += (iteration_weight * player_reach) * strategy


def average_strategy(strat_sum: np.ndarray) -> np.ndarray:
    """Normalize accumulated strategy sums to a valid policy."""
    s = float(strat_sum.sum())
    if s <= 0.0:
        return np.full_like(strat_sum, 1.0 / strat_sum.size)
    return strat_sum / s
