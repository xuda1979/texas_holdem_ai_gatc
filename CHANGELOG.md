# Changelog

All notable changes to Texas Hold'em AI GATC are documented here. The project
follows [semantic versioning](https://semver.org/) and the format here is
loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- MIT `LICENSE` file (previously referenced by README but missing from the
  repository).
- `CONTRIBUTING.md` describing the development setup and PR checklist.
- `scripts/` and `memory/` directories are now tracked in git (they were
  previously `.gitignore`d, which hid user-facing Huanxin launch helpers and
  the project journal from clones).
- Ruff `[lint]` section in `ruff.toml` (the top-level form is deprecated).

### Changed
- `src/poker_ai/gui/simulation.py` now imports `PokerGameGUI` from
  `poker_ai.gui` (previously referenced an undefined name).
- Import sorting fixed across the codebase via `ruff --fix --select I001`.

### Fixed
- Removed 27 unused imports and 4 unused local variables flagged by ruff.
- Modernized deprecated ruff configuration so `ruff check` no longer warns.

## [0.1.0] - 2026-04-09

### Added
- Initial release of the Texas Hold'em AI GATC package.
- CFR, CFR+, and Deep CFR trainers with reservoir replay.
- Kuhn poker reference implementation and exploitability benchmark.
- FastAPI service (`app.py`) exposing `/healthz`, `/readyz`, `/metrics`, and
  `/play` endpoints.
- Tkinter GUI for human-vs-AI play.
- CLI entry points (`play-poker`, `train-poker`, `poker-ai-gcp-train`).
- GitHub Actions workflows for build/test, e2e tests, perf smoke, and secret
  scanning.
