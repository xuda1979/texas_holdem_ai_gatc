# 2026-07-24 — Polyak-averaged Target Network for SingleNetworkCFRTrainer

## Summary

Addressed the **regression-after-1000-samples** pattern documented in
`SESSION_20260713_TRAJECTORY.md`. The root cause is a moving-target
instability in neural CFR: as the advantage network updates, the regret
baseline (state-value) used to compute `r(a) = Q(a) - V` shifts, so the
network is chasing a non-stationary target. The standard fix (DQN-style)
is a Polyak-averaged target network that moves slowly, anchoring the
baseline.

## Changes

- `src/poker_ai/ai/trainers/single_network_cfr_trainer.py`:
  - New `target_ema` ctor parameter (default `1.0` = legacy, `0.995`
    recommended).  `τ = 1.0` disables the target network.
  - `target_model` (frozen `AdvantageNetwork` copy) initialised from
    the online network; parameters marked `requires_grad_(False)`.
  - In `train_step`, when `target_ema < 1.0` the regret baseline blends
    the CFR regret-matched policy `σ_t` with the target network's
    softmax policy: `σ_blend = (1-τ)·σ_t + τ·σ_target`.  The online
    network still receives the gradient; only the baseline is anchored.
  - After `optimizer.step()`, EMA update:
    `θ_target ← τ·θ_target + (1-τ)·θ_online`.
  - `load_model` now syncs the target network so resumption starts from
    a stable baseline.
  - `config` dict records `target_ema` and `iteration_weight_exponent`.
- `src/poker_ai/cli/train.py`: reads `model.target_ema` from config and
  passes it to the trainer.
- `poc_ai3_npu.yaml`: documents and sets `target_ema: 0.995`.
- `tests/test_single_network_cfr_trainer.py`: 4 new tests
  - `test_target_network_initialised_and_synced`
  - `test_target_network_ema_updates_after_train_step`
  - `test_target_ema_disabled_matches_legacy`
  - `test_load_model_syncs_target_network`

## Test Results

```
355 passed, 8 skipped (was 351 passed, 8 skipped)
mypy: clean on modified files (0 errors / 2 files)
```

## Why this matters for "world best"

The single biggest obstacle to a competitive neural CFR agent is
training stability — the model peaks early then degrades.  This fix
directly targets that failure mode.  Combined with the existing
`iteration_weight_exponent` (sample weighting) and reservoir-sampled
replay buffer, the trainer now has the three standard stability
mechanisms from the Deep CFR / ReBeL literature.

## Next steps

- Once Huanxin auth is restored, launch a run with `target_ema: 0.995`
  and confirm the peak at ~1000 samples is preserved rather than
  regressing.
- Build a real local best-response exploitability evaluator for the
  abstracted Texas Hold'em game tree (the current
  `evaluation/exploitability.py` is matrix-game only).  This is the
  missing convergence metric needed to claim Nash-equilibrium quality.
