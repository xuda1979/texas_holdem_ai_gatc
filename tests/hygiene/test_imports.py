from __future__ import annotations

from pathlib import Path


def test_no_shadow_top_level_modules() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src" / "poker_ai"

    top_level: set[str] = {p.stem for p in repo_root.glob("*.py")}
    top_level |= {
        p.name for p in repo_root.iterdir() if p.is_dir() and (p / "__init__.py").exists()
    }

    src_modules: set[str] = {p.stem for p in src_root.glob("*.py")}
    src_modules |= {p.name for p in src_root.iterdir() if p.is_dir()}

    # Intersection indicates modules at the repo root shadow those in src/poker_ai
    shadowed = top_level & src_modules
    assert not shadowed, f"Top-level modules shadow poker_ai modules: {sorted(shadowed)}"
