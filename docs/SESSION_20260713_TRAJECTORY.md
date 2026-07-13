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

## Updated Trajectory (run2, 300 hands each)

| samples | bb/100 vs calling_station | stderr |
|---------|---------------------------|--------|
| 396     | -212.07                   | 107    |
| 747     | +77.57                    | 105    |
| 1002    | +168.00 (NAS best)        | 112    |
| 1088    | +111.30 (peak, in-run)    | 110    |
| 1294    | +21.83                    | 76     |
| 1400    | -175.63                   | 155    |
| 1650    | -97.57                    | 100    |

Regression onset: between 1088 and 1294. By 1400 deeply negative.

## Weight Norm Diagnostic
Checked |advantage_net| and |policy_net| L2 norms at samples 100 and 1650:
both ~70.6 with <0.1% drift. **No catastrophic weight explosion** -- the
regression is a subtle policy-quality issue, not numerical instability.

## Root Cause Analysis (deeper)

The self-play is **pure self-play** (both players use the same model).
The model converges toward a Nash equilibrium of self-play, which is NOT
the same as an exploitive strategy vs calling_station. Early in training
(samples 400-1100), the model passes through a transient state that
happens to exploit calling_station well. As training continues, self-play
convergence "unlearns" this exploitive behavior in favor of the Nash
equilibrium, which is more passive/non-exploitive against a station.

Key evidence:
- The strategy buffer records the epsilon-mixed policy at each infoset
  (self_play.py line 731). As the advantage net converges to Nash, the
  recorded strategies become less exploitive.
- Weight norms are stable, so this is not a training instability -- it
  is the optimization objective (self-play Nash) diverging from the
  evaluation target (exploit calling_station).

## Fix Implemented This Session: Best-Checkpoint Tracking
- `TrainingEvalHook` now saves the peak checkpoint (vs `best_baseline`)
  to `models/best/` with a `_latest` symlink.
- Optional `stop_on_regression` halts training after sustained regression
  (margin + patience).
- `evaluation.every_samples` lowered 1000 -> 500 for finer granularity.
- Run3 started at 2026-07-13 06:01 UTC with these features enabled.

## Next Fix Proposal: Add Calling Station as Training Opponent
To make the model ROBUSTLY exploit calling_station (not just transiently),
the self-play should mix in calling_station as an opponent during a
fraction of traversals. This gives the advantage net direct signal about
calling_station's exploitable tendencies (calling too much, never folding),
so the Nash equilibrium of the mixed game includes exploitation of the
station. This is "fictitious play" / opponent-pool self-play.

## 2026-07-13 Update: Opponent-Pool Self-Play Implemented

### Run6 Status (in progress)
Run6 (`ai3-rd-20260713T074300Z`) restarted from the run2 peak checkpoint
`deep_cfr_time_1088.pth` (+111 bb/100 vs calling_station) with the existing
fixes (lr 3e-4, buffer 5M, epsilon 0.05, best-checkpoint tracking, eval
every 500 samples).

**Milestone at 512 samples:** new peak **+116.55 bb/100** vs calling_station
(up from run2's +111.30). The restart from the peak checkpoint held and
improved. No regression detected through 864 samples (cycle 28).
Best checkpoint: `models/best/best_calling_station_512.pth`.

### Opponent-Mix Feature (implemented, not yet deployed)
Added `self_play.opponent_mix` config to `SelfPlay`:
- `probability` (float 0-1): per-hand probability that one non-traverser
  seat plays a fixed baseline strategy instead of the model policy.
- `strategy` (str): the fixed strategy name. Currently supports
  `calling_station` (default) and `always_fold`.

Mechanism: on a "mixed-opponent" hand, the designated non-traverser seat
plays the fixed strategy (e.g. calling_station check/call) while the
traverser still collects regret/strategy samples via the normal MCCFR
traversal. The strategy-buffer sample at the opponent node still uses the
*model's* policy (preserving Deep CFR average-strategy semantics), but the
*action taken along the trajectory* is the fixed strategy's action. This
gives the advantage network direct exploitation signal: "what's the value
of each action when the opponent plays calling_station?".

Config (in `poc_ai3_npu.yaml`):
```yaml
self_play:
  opponent_mix:
    probability: 0.3
    strategy: calling_station
```

Tests: `tests/selfplay/test_opponent_mix.py` (9 tests, all passing).
Covers config parsing/clamping, unknown-strategy handling, flat-config
fallback, action-index mapping for calling_station/always_fold, and a full
`play_hand_for_training` smoke test with the feature enabled.

### Next Steps
- Let run6 continue to confirm whether the regression still occurs around
  1000-1100 samples with the existing fixes.
- When run6 regresses or completes, deploy run7 with `opponent_mix.probability=0.3`
  to test whether fictitious-play self-play produces a robustly positive
  calling_station result that doesn't regress.
- If 0.3 is too disruptive, try 0.1-0.2. If still regressing, try 0.5.

