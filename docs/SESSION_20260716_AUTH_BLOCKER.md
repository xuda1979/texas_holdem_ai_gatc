# 2026-07-16 — Huanxin Auth Blocker + Local Verification

## Summary
Attempted to continue R&D and launch run7 (opponent-mix self-play) on ai3, but
Huanxin authentication has expired and cannot be refreshed without human
intervention. Used the blocked remote window to verify the local codebase is
production-ready and the run7 launch path is correct.

## Huanxin Auth Status — BLOCKED (needs human action)
- Daemon `ai3` startup fails: `authDriftDetected: true`, `startupState: error`.
- API-key smoke returns `401 Unauthorized` — the stored API keys in
  `~/.codex/secrets/huanxin.env` are stale/expired.
- Safari SSO bridge (`HUANXIN_ALLOW_SAFARI_SSO_BRIDGE=1`) was attempted but the
  Safari session is also at `login_required` (Keycloak login page rendered).
- No `HUANXIN_LOGIN_PHONE` / `HUANXIN_LOGIN_PASSWORD` are stored anywhere on
  disk, so `huanxin_auto_login.js` password fallback cannot run.
- **Action required from human**: log into https://aihuanxin.cn in Safari
  (foreground, normal browsing). Once the Safari session is live, the SSO
  bridge will be able to refresh the headless daemon and training operations
  can resume. Alternatively, provide phone+password credentials in
  `~/.codex/secrets/huanxin.env` as `HUANXIN_LOGIN_PHONE` / `HUANXIN_LOGIN_PASSWORD`.

## Local Verification (project `.venv`, Python 3.11)
- Full core test suite: **353 passed, 2 skipped, 0 failures**.
  - Skips: `tests/e2e/test_play.py` (needs `E2E_BASE_URL`),
    `tests/test_self_play.py` CUDA subtest (no local CUDA).
- Opponent-mix self-play tests: **9 passed** (`tests/selfplay/test_opponent_mix.py`).
- Note: tests MUST be run with `/Users/daxu/software/cfr/.venv/bin/python` —
  the system `python3` is missing `eval7` (a declared dependency in
  `requirements.txt` / `pyproject.toml`). Without `eval7`,
  `test_showdown_awards_blinds` fails because treys cannot evaluate a hand
  with an empty/partial board and the fallback import raises.

## Run7 Launch Path — Verified Correct
`scripts/ai3_launch_run7_opponent_mix.sh`:
1. Copies `models/best/best_calling_station_latest.pth` → `models/run7_start.pth`.
2. Launches `run_training.py --config poc_ai3_npu.yaml --algorithm deep_cfr --npus`.
3. Config `self_play.opponent_mix` is active: `probability=0.3, strategy=calling_station`.
4. `best_calling_station_latest.pth` is maintained as a symlink by
   `_save_best_checkpoint` (`src/poker_ai/evaluation/comprehensive.py:1011`),
   so the `cp -f` will resolve the symlink and copy the peak checkpoint.
5. Starting point: run6 best (`best_calling_station_512.pth`, +116.55 bb/100
   vs calling_station, per MEMORY.md).

## Current Training State (last known, from 2026-07-13)
- Run6 confirmed the calling_station regression pattern: model peaks ~512
  samples (+116 bb/100) then regresses to -132 bb/100 by 1024 samples
  (commit `cfe81a7a`).
- Run7 (opponent-mix) is the next experiment to test whether fictitious-play
  self-play produces a robustly positive calling_station result that doesn't
  regress.
- Run7 has NOT been launched yet — blocked by the auth issue above.

## Next Steps (once auth restored)
1. Re-run `scripts/ai3_shell.sh "ls -la /root/work/software/cfr/models/best/"`
   to confirm run6's final best checkpoint on ai3.
2. Launch run7: `bash scripts/ai3_launch_run7_opponent_mix.sh`.
3. Monitor first 512 samples — if calling_station stays positive past 1024
   samples, the opponent-mix fix works.
4. Back up important ai3 artifacts to NAS (`/root/work/backups/cfr/`):
   - `models/best/best_calling_station_*.pth`
   - `models/deep_cfr_time_*.pth` (peak)
   - `logs/*.log`
   - `models/eval_*.json`
