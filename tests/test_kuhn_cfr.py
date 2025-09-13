from poker_ai.evaluation.kuhn_cfr import train_kuhn_cfr


def test_kuhn_bet_frequencies_monotone():
    """
    Qualitative Kuhn Poker signal:
    Optimal-like behavior should satisfy bet(J) < bet(Q) < bet(K).
    This is a robust monotonicity check rather than strict numeric targets.
    """
    k = train_kuhn_cfr(iterations=15_000, seed=123)
    j, q, kbet = k.root_bet_probs()

    # sanity: each is within [0,1]
    for v in (j, q, kbet):
        assert 0.0 <= v <= 1.0

    # monotone: J < Q < K
    assert j < q < kbet

    # also, strategies should not be degenerate at root
    assert j < 0.5 and kbet > 0.5

