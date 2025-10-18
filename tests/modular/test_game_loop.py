from gatc_modular.testing.fakes import CountingGameEngine, FirstLegalPolicy
from gatc_modular.services.game_loop import GameLoop


def test_episode_runs_and_finishes() -> None:
    eng = CountingGameEngine(horizon=4)
    loop = GameLoop(eng, {0: FirstLegalPolicy(), 1: FirstLegalPolicy()})
    result = loop.play_episode(seed=1)
    assert result.steps == 4
    # In the fake engine, winner may be 0, 1, or None depending on final action.
    assert result.winner in (0, 1, None)
    assert set(result.total_reward.keys()) == {0, 1}


def test_illegal_action_is_auto_corrected_when_enforced() -> None:
    class BadPolicy(FirstLegalPolicy):
        def select_action(self, obs, legal_actions, player_id):  # type: ignore[override]
            return 99  # definitely illegal

    eng = CountingGameEngine(horizon=2)
    loop = GameLoop(eng, {0: BadPolicy(), 1: FirstLegalPolicy()}, enforce_legal_actions=True)
    result = loop.play_episode()
    assert result.steps == 2


def test_illegal_action_raises_when_not_enforced() -> None:
    class BadPolicy(FirstLegalPolicy):
        def select_action(self, obs, legal_actions, player_id):  # type: ignore[override]
            return 99

    eng = CountingGameEngine(horizon=1)
    loop = GameLoop(eng, {0: BadPolicy(), 1: FirstLegalPolicy()}, enforce_legal_actions=False)
    try:
        loop.play_episode()
        assert False, "Expected ValueError for illegal action"
    except ValueError:
        pass
