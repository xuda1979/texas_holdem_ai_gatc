from gatc_modular.testing.fakes import CountingGameEngine, FirstLegalPolicy
from gatc_modular.services.self_play import SelfPlayService


def test_self_play_stats_aggregate() -> None:
    eng = CountingGameEngine(horizon=3)
    svc = SelfPlayService(eng, {0: FirstLegalPolicy(), 1: FirstLegalPolicy()})
    stats = svc.run(n_episodes=5, seed=123)
    assert stats.episodes == 5
    # With FirstLegalPolicy always taking action 0, the fake engine never awards a winner.
    assert all(w == 0 for w in stats.wins.values())
    assert stats.avg_steps == 3.0
    assert stats.total_reward[0] == 0.0
    assert stats.total_reward[1] == 0.0
