from .exploitability import calculate_exploitability, compute_best_response
from .performance_analysis import ModelPerformanceAnalyzer, run_tournament

# ``ai_gto_analyzer`` depends on optional packages (e.g., PyYAML).  Import it
# lazily so that basic evaluation utilities remain available in minimal test
# environments.
try:  # pragma: no cover - executed only when optional deps are installed
    from .ai_gto_analyzer import display_ai_gto_stats
except Exception:  # pragma: no cover - optional feature missing
    display_ai_gto_stats = None

__all__ = [
    'calculate_exploitability',
    'compute_best_response',
    'ModelPerformanceAnalyzer',
    'run_tournament',
    'display_ai_gto_stats'
]
