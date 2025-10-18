from gatc_modular.testing.fakes import CountingGameEngine
from gatc_modular.ports.engine import Engine


def test_engine_protocol_is_satisfied() -> None:
    eng = CountingGameEngine(horizon=3)
    # runtime_checkable allows isinstance() on Protocols
    assert isinstance(eng, Engine)
    obs = eng.reset(seed=42)
    # observation shape from fake engine is (t, current_player)
    assert tuple(obs) == (0, 0)
    assert eng.legal_actions() == [0, 1]
    assert not eng.is_terminal()
    assert eng.current_player() == 0
    # cloning should preserve configuration
    assert isinstance(eng.clone(), CountingGameEngine)
