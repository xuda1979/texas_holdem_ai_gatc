# Huanxin ai3 Training — 2026-07-13 Regression Trajectory Mapping

## Summary
The 2026-07-12 restart (lr 3e-4, buffer 5M, epsilon 0.05) exhibits the SAME
regression pattern as the previous run: the model learns to beat
calling_station, peaks around 1000-1100 samples, then degrades. The fixes
delayed but did not prevent the regression.

## Current Run Trajectory (300 hands each vs calling_station)

| samples | bb/100  | stderr | verdict          |
|---------|---------|--------|------------------|
| 396     | -212.07 | 107    | losing (early)   |
| 747     | +77.57  | 105    | winning          |
| 1088    | +111.30 | 110    | **peak**         |
| 1650    | -97.57  | 100    | regressed        |

Training is at cycle 53 / ~1664 samples as of 2026-07-13 02:22 UTC.
Training process (PID 2734) still running, loss ~1.39, replay buffer 580K.

## Benchmark (Previous Run, NAS-backed-up)

| checkpoint                         | bb/100 vs calling_station (300h) |
|------------------------------------|----------------------------------|
| best_calling_station_1002.pth      | +168.00 ± 112.00                 |

The previous run's peak at sample 1002 (+168) is STRONGER than the current
run's peak at 1088 (+111). The "fixes" may have slightly weakened peak
performance while also not preventing regression.

## All baselines at current peak (ckpt 1088, 300 hands)
- always_fold:    +75.00 ± 0.00
- calling_station:+111.30 ± 109.73
- random_uniform: +584.23 ± 183.37

## Root Cause Hypotheses (to investigate)
1. **Policy network catastrophic forgetting**: The policy net is trained on
   the strategy buffer, which fills with recent self-play. As the advantage
   net drifts, the strategy buffer gets contaminated with degenerate strategies.
2. **Replay buffer not diverse enough**: Even at 5M capacity, reservoir
   sampling may evict the diverse early samples that contained good
   calling_station counter-strategies.
3. **Regret matching collapse**: As advantages saturate/oscillate, the
   regret-matched policy may collapse to a single action (e.g., all-in),
   which loses to calling_station (never folds).
4. **No opponent modeling**: Self-play only sees itself; calling_station is
   a very different opponent profile. The model may over-specialize to
   self-play equilibrium.
5. **Learning rate still too high**: 3e-4 may still overshoot after the
   advantage net finds a good basin.

## Next Steps
- Eval more checkpoints between 1088 and 1650 to pin the exact regression
  onset in this run.
- Inspect advantage_net vs policy_net weight norms across checkpoints.
- Consider: (a) lower LR (1e-4), (b) freeze policy net periodically,
  (c) add calling_station as a training opponent, (d) use best-checkpoint
  tracking that evaluates vs calling_station and saves the best.
- Back up the current peak checkpoint (1088) to NAS.
