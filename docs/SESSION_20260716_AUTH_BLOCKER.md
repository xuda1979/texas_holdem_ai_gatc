# 2026-07-16 — Huanxin Auth Blocker + Local Verification

## Summary
Attempted to continue R&D and launch run7 (opponent-mix self-play) on ai3, but
Huanxin authentication has expired and cannot be refreshed without human
intervention. Used the blocked remote window to verify the local codebase is
production-ready, harden run7 diagnostics, and add a regression-analysis tool.

## Work Completed This Session (3 commits)
1. **`b6c98657` docs: auth blocker + local verification**
   - Verified full core test suite passes with project `.venv`: 353 tests.
   - Verified run7 launch path resolves `best_calling_station_latest.pth` correctly.
   - Documented the auth blocker and the human action required to unblock.
   - Tracked `.dvc/config` (Huanxin S3 remote for NAS backups).

2. **`e757ffb4` feat: checkpoint norm analysis tool**
   - `tools/analyze_checkpoint_norms.py`: computes L2 norms and max-abs weights
     for advantage_net and policy_net across the training trajectory.
   - Directly addresses the "inspect advantage_net vs policy_net weight norms"
     next-step from `SESSION_20260713_TRAJECTORY.md`.
   - Pure CPU, read-only — safe to run on ai3 while training is live.
   - 6 unit tests (synthetic checkpoints, legacy bare-state-dict, sorting).

3. **`73af0f26` feat: opponent-mix engagement logging**
   - Added cumulative counters to `SelfPlay` tracking how many hands used the
     calling_station fixed-opponent vs total hands.
   - Logs engagement rate every 500 hands so run7 results can be verified to
     actually engage the fixed-opponent strategy at the configured probability.
   - 2 new tests (engagement with prob=1.0 and prob=0.0).

**Final test state: 361 passed, 2 skipped, 0 failures** (project `.venv`).

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
2. **Sync the new tool + instrumentation to ai3** before launching run7:
   `scripts/ai3_shell.sh "cd /root/work/software/cfr && rsync -a --delete src/ /root/work/software/cfr/src/"` (or push via the sync script).
3. Launch run7: `bash scripts/ai3_launch_run7_opponent_mix.sh`.
4. Monitor first 512 samples — if calling_station stays positive past 1024
   samples, the opponent-mix fix works. Watch the log for
   `opponent-mix engagement | hands=... | rate=...` lines to confirm the
   fixed-opponent strategy is actually being used at ~0.3 rate.
5. **Run the checkpoint norm analysis** on run6's existing checkpoints to
   retroactively diagnose the regression:
   `scripts/ai3_shell.sh "cd /root/work/software/cfr && python3 tools/analyze_checkpoint_norms.py --checkpoint-dir models --pattern 'deep_cfr_time_*.pth' --output models/checkpoint_norms.json --include-best"`
6. Back up important ai3 artifacts to NAS (`/root/work/backups/cfr/`):
   - `models/best/best_calling_station_*.pth`
   - `models/deep_cfr_time_*.pth` (peak)
   - `logs/*.log`
   - `models/eval_*.json`
   - `models/checkpoint_norms.json` (from step 5)
