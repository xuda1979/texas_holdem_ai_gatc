# Huanxin ai3 Training — 2026-07-12 Fixes and Restart

## Summary
Previous training run (`ai3-rd-20260711T141826Z`) regressed on calling_station
after sample 1002. Stopped the run, diagnosed root causes, applied fixes, and
restarted training.

## Regression Diagnosis (Previous Run)

| samples | always_fold | calling_station | random_uniform |
|---------|-------------|-----------------|----------------|
| 506     | +75.0       | +24.73          | +756.43        |
| 1002    | +75.0       | **+66.17**      | +870.57        |
| 1510    | +72.33      | **-115.07**     | +505.33        |
| 2003    | +74.33      | **-166.33**     | +540.92        |
| 2500    | +75.0       | **-126.40**     | +846.33        |
| 2838    | +75.0       | **-35.54**      | +540.92        |
| 3222    | +74.33      | **-127.43**     | +313.60        |

## Root Causes
1. **Learning rate too high (1e-3)**: Loss plateaued at ~1.3-1.5 by cycle 11
   and never refined. The model bounced around a local minimum.
2. **Replay buffer too small (1M)**: Buffer filled at cycle 93 (~2800 samples)
   then evicted old diverse self-play data, reinforcing the current degenerate
   strategy.
3. **Epsilon-greedy exploration not applied during MCCFR**: The
   `self_play.epsilon: 0.05` config was only read by the standalone CLI
   self-play (`cli/self_play.py`), NOT by the trainer's MCCFR traversal
   (`selfplay/self_play.py`). Pure regret matching collapsed to over-aggressive
   lines.

## Fixes Applied (commit bda44279)
1. `poc_ai3_npu.yaml`: `model.learning_rate: 3.0e-4` (was 1e-3 default)
2. `poc_ai3_npu.yaml`: `model.buffer_capacity: 5000000` (was 1M default)
3. `src/poker_ai/cli/train.py`: Pass `buffer_capacity` to DeepCFRTrainer;
   merge `self_play` config section into training_params handed to SelfPlay
   manager.
4. `src/poker_ai/selfplay/self_play.py`: Read `epsilon` from training_config;
   apply epsilon-greedy mixing (regret-matched policy + uniform over legal
   actions) in `_get_policy`.

## Verification
- All 83 existing tests pass (8 skipped for optional deps).
- Smoke test confirms `replay_buffer.capacity = 5000000`,
  `optimizer.param_groups[0]['lr'] = 0.0003`, `SelfPlay.epsilon = 0.05`.
- New training log confirms: `SelfPlay epsilon-greedy exploration enabled |
  epsilon=0.050`.

## New Training Run
- Job: `ai3-rd-20260712T142600Z` (PID 2734 on ai3)
- Log: `logs/ai3-rd-20260712.log`
- Config: `poc_ai3_npu.yaml` (with fixes)
- Started: 2026-07-12 14:26 UTC
- First checkpoint: `deep_cfr_time_38.pth` at 14:36 (10 min for first cycle)

## Early Evaluation (100 samples)
| baseline       | bb/100  | stderr  |
|----------------|---------|---------|
| always_fold    | +74.00  | ±1.00   |
| calling_station| -171.95 | ±137.41 |
| random_uniform | +958.30 | ±195.98 |

At 100 samples, calling_station is very negative — expected at this early
stage (previous run was -425 at 288 samples). Need to wait for 500-1000
samples to see if the fixes prevent the regression.

## NAS Backup
- Snapshot: `/root/work/backups/texas-holdem/snapshot-20260712T1419Z.tar.gz`
  (39KB — config, code, evals, logs)
- Key checkpoints: `/root/work/backups/texas-holdem/checkpoints/`
  - `best_calling_station_1002.pth` (19MB) — best vs calling_station
  - `early_506.pth` (19MB) — early checkpoint
  - `latest_before_fix_2838.pth` (19MB) — last before fixes
- Previous run checkpoints archived to `models/run1/` on ai3 (249 files).

## Next Steps
- Wait for 500-1000 samples, evaluate, compare to previous run.
- If calling_station stays positive at 1000+ samples, fixes worked.
- If calling_station still regresses, consider further changes:
  - Even lower learning rate (1e-4)
  - Larger epsilon (0.1)
  - Add opponent modeling (random baseline opponents during self-play)
  - Use single_network_cfr_trainer instead of deep_cfr
