# 2026-07-17 — Publish-Readiness Work (Huanxin Auth Still Blocked)

## Summary

Huanxin auth is still expired (API-key smoke returns 401 Unauthorized), so
run7 still cannot be launched. Used the blocked window to advance
publish-readiness of the local codebase. All changes below are verified by
the full test suite (378 passed, 2 skipped) and a clean wheel build.

## Changes

### Packaging metadata
- `pyproject.toml`: added `readme`, `license`, `authors`, `keywords`,
  `classifiers`, and `[project.urls]` (homepage/repository/issues/changelog).
  Wheel builds clean: `texas_holdem_ai_gatc-0.1.0-py3-none-any.whl` with
  LICENSE included in `dist-info/licenses/`.

### Missing OSS files
- `LICENSE`: MIT, copyright 2024-2026 contributors (README previously
  pointed to a file that did not exist).
- `CONTRIBUTING.md`: development setup, test/lint commands, PR checklist.
- `CHANGELOG.md`: Unreleased section + 0.1.0 baseline.

### Repository hygiene
- `.gitignore`: stopped ignoring `scripts/` and `memory/`. `scripts/`
  contains user-facing Huanxin launch helpers (`ai3_shell.sh`,
  `ai3_launch_run7_opponent_mix.sh`, etc.) that clones need to reproduce
  the training workflow. `memory/` is the project journal referenced from
  `AGENTS.md`. Binary artifacts (`models/`, `logs/`, `.huanxin_jobs/`,
  `.venv/`) remain ignored.

### Documentation fixes
- `README.md`: replaced `OWNER` placeholder with the actual GitHub owner
  (`xuda1979`) in the CI badge and clone command. Fixed `tools/verify_images.py`
  → `verify_images.py` (script lives at repo root). Fixed
  `configs/self_play.yaml` → `src/poker_ai/config/config.yaml` in the GCE
  launcher example.
- `docs/cloud_training_guide.md`: same `OWNER` → `xuda1979` fix in three
  clone commands.

### Lint cleanup
- Removed 27 unused imports and 4 unused local variables.
- Fixed `PokerGameGUI` undefined-name in `src/poker_ai/gui/simulation.py`
  (it was referenced without an import; now imported from `poker_ai.gui`).
- Removed dead `N = len(contribs)` in `src/poker_ai/rules/side_pots.py`.
- Modernized `ruff.toml` to use the `[lint]` section (top-level
  `select`/`ignore`/`isort` are deprecated in current ruff).
- Auto-fixed import sorting (I001) across the codebase.

## Verification
- `pytest -q`: 378 passed, 2 skipped (1 e2e needs `E2E_BASE_URL`, 1 needs
  CUDA), 1 warning (expected — `play_vs_ai.py` falls back to
  `RandomAIStrategy` when given a dummy model path).
- `python -m build --wheel`: succeeds, METADATA includes the new
  authors/license/classifiers/urls.
- `ruff check --select F`: all checks pass.

## Huanxin Blocker (unchanged)
- `scripts/ai3_shell.sh "echo ok"` fails with
  `Huanxin auth expired before opening ai3; API-key smoke={"ok":false,"status":401}`.
- Daemon on port 19003 was killed to clear a stale lock; subsequent
  retries still hit the 401.
- Without stored `HUANXIN_LOGIN_PHONE`/`HUANXIN_LOGIN_PASSWORD` env vars or
  a human logging into aihuanxin.cn in Safari, this is a hard blocker for
  remote training/eval and NAS backup.

## Next Steps
- **Publish-readiness**: address remaining `mypy` errors (135, mostly
  pre-existing attr-defined and arg-type in `cli/play.py`, `gui/gui.py`,
  `utils/abstraction.py`). Add a `tests/e2e/test_health.py` smoke test.
- **Once Huanxin auth is restored**:
  1. Verify run6's best checkpoint on ai3 (`models/best/best_calling_station_512.pth`).
  2. Launch run7 via `scripts/ai3_launch_run7_opponent_mix.sh`.
  3. Monitor calling_station bb/100 past 1024 samples.
  4. Back up ai3 artifacts to NAS (`/root/work/backups/cfr/`).
