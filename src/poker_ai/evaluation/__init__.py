from typing import TYPE_CHECKING, Any

from .exploitability import calculate_exploitability, compute_best_response

# ``performance_analysis`` pulls in a large portion of the training stack which
# is optional for many lightweight environments.  Import it lazily so basic
# helpers like ``calculate_exploitability`` remain usable even when the full
# dependency tree is unavailable.
try:  # pragma: no cover - executed only when optional deps are installed
    from .performance_analysis import ModelPerformanceAnalyzer, run_tournament
except Exception:  # pragma: no cover - optional feature missing
    if TYPE_CHECKING:
        from .performance_analysis import ModelPerformanceAnalyzer, run_tournament  # noqa: F811

    ModelPerformanceAnalyzer = None  # type: ignore[assignment,misc]

    def run_tournament(  # type: ignore[no-redef]
        model_paths: list[str], games_per_match: int = 10, device: str = "cpu"
    ) -> dict[str, int]:
        raise RuntimeError("performance analysis utilities unavailable")

# ``ai_gto_analyzer`` depends on optional packages (e.g., PyYAML).  Import it
# lazily so that basic evaluation utilities remain available in minimal test
# environments.
try:  # pragma: no cover - executed only when optional deps are installed
    from .ai_gto_analyzer import display_ai_gto_stats
except Exception:  # pragma: no cover - optional feature missing
    display_ai_gto_stats = None  # type: ignore[assignment]

__all__ = [
    "calculate_exploitability",
    "compute_best_response",
    "ModelPerformanceAnalyzer",
    "run_tournament",
    "display_ai_gto_stats",
]

