from .kuhn_poker import KuhnCFR


def train_kuhn_cfr(iterations: int = 20_000, seed: int = 7):
    """
    Trains CFR on Kuhn Poker and returns the trainer.
    Useful as a tiny convergence sanity test harness.
    """
    k = KuhnCFR()
    k.train(iterations=iterations, seed=seed)
    return k

