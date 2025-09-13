import random
import numpy as np


def test_long_run_expected_value_near_zero():
    rng = random.Random(0)
    payoffs = [rng.choice([-1, 1]) for _ in range(200)]
    mean_diff = sum(payoffs) / len(payoffs)
    boot = []
    for _ in range(500):
        sample = [rng.choice(payoffs) for _ in range(len(payoffs))]
        boot.append(sum(sample) / len(sample))
    lower, upper = np.percentile(boot, [2.5, 97.5])
    assert lower <= 0 <= upper
    assert abs(mean_diff) < 0.1
