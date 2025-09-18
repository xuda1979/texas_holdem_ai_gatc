import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "gatc_poker" / "pots.py"
    spec = importlib.util.spec_from_file_location("_gatc_pots_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_gatc_pots_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_folded_player_extra_contribution_attaches_to_last_live_pot():
    pots_module = _load_module()
    pots = pots_module.compute_side_pots({0: 100, 1: 50, 2: 200}, {0, 1})
    assert [p.amount for p in pots] == [150, 200]
    assert [set(p.eligible) for p in pots] == [{0, 1}, {0}]


def test_zero_contributions_or_no_showdown_returns_empty():
    pots_module = _load_module()
    assert pots_module.compute_side_pots({}, {0}) == []
    assert pots_module.compute_side_pots({0: 100}, set()) == []
