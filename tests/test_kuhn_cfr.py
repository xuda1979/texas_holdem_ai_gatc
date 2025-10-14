import random

from cfr_algorithm.kuhn_cfr import KuhnCFR


def test_kuhn_cfr_converges_value():
    random.seed(0)
    cfr = KuhnCFR()
    cfr.train(iterations=60000)
    avg = cfr.average_strategy()
    ev = KuhnCFR.expected_value_p1(avg)
    target = -1.0 / 18.0  # known game value for player 1 at equilibrium
    assert abs(ev - target) < 0.02, f"EV {ev} deviates too much from Kuhn equilibrium {target}"
