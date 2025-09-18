from poker_ai.evaluation.kuhn_cfr import train_kuhn_cfr
from poker_ai.evaluation.kuhn_poker import KuhnCFR


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


def test_terminal_state_detection():
    base = "JK"
    assert not KuhnCFR._is_terminal(base)
    assert not KuhnCFR._is_terminal(base + "p")
    assert not KuhnCFR._is_terminal(base + "b")
    assert not KuhnCFR._is_terminal(base + "pb")
    assert KuhnCFR._is_terminal(base + "pp")
    assert KuhnCFR._is_terminal(base + "bp")
    assert KuhnCFR._is_terminal(base + "pbp")
    assert KuhnCFR._is_terminal(base + "bb")
    assert KuhnCFR._is_terminal(base + "pbb")


def test_terminal_utility_signs():
    assert KuhnCFR._terminal_utility_p1("JKbp") == 1  # P1 bet, P2 folded
    assert KuhnCFR._terminal_utility_p1("JKpbp") == -1  # P2 bet, P1 folded
    assert KuhnCFR._terminal_utility_p1("KJpp") == 1
    assert KuhnCFR._terminal_utility_p1("JKpp") == -1
    assert KuhnCFR._terminal_utility_p1("KJbb") == 2
    assert KuhnCFR._terminal_utility_p1("JKbb") == -2

