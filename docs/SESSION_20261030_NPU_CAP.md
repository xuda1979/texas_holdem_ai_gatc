# Session 2026-10-30: NPU per-environment cap (≤8) + real Kuhn exploitability

## Context

Two work items this session:

1. Memory `2026-07-24.md` "Next" item: a real best-response exploitability
   evaluator.  Leduc already has one (commit `eb2bddb1`); the remaining gap
   was that `evaluation/comprehensive.py` reported a **synthetic** Kuhn
   exploitability (0.0/1.0 flag) instead of the real metric.
2. User directive: *"only use not more than 8 NPUs in one environment in
   ASI1,2,3"*.  The ASI machines are shared; a job must never monopolise
   every visible chip.

## Changes

### 1. Real Kuhn exploitability in the comprehensive evaluator

- `src/poker_ai/evaluation/kuhn_cfr.py`
  - New `kuhn_strategy_table(trainer)`: converts the legacy `KuhnCFR`
    average strategy (letter cards `J/Q/K`, `{"p":…,"b":…}` dicts) into the
    exact-evaluator format (int cards `1/2/3`, `[prob_pass, prob_bet]`
    lists keyed `"{card}{history}"`).
  - New `kuhn_exploitability(trainer)`: exact BR exploitability via
    `poker_ai.games.kuhn.exploitability`.
- `src/poker_ai/evaluation/comprehensive.py`
  - `_compute_kuhn_exploitability` now reports the **real** value;
    `converged` ⇔ `eps ≤ 0.05`.
  - Sanity numbers: 50k iters → eps ≈ 0.0042; 200 iters → eps ≈ 0.13;
    uniform → 0.458.
- Tests: `tests/evaluation/test_kuhn_cfr_exploitability.py` (5 tests:
  table format, agreement with legacy nodes, monotone convergence,
  agreement with the exact evaluator, uniform baseline).
- `tests/eval/test_comprehensive_evaluation.py`: the kuhn-enabled test now
  asserts `0 < eps < 1` (a real number, not the old synthetic flag).

### 2. ≤8 NPUs per environment on ASI1/2/3

Layered enforcement (belt and braces):

| Layer | Mechanism | Effect |
|---|---|---|
| Launch scripts | `scripts/npu_limit.sh` (new, sourced by all ASI train scripts) | `ASI{1,2,3}_NUM_NPUS` (default 1, hard cap 8) → `ASCEND_RT_VISIBLE_DEVICES` exported in the remote command; requests > 8 abort with exit 1 |
| Python CLI | `MAX_NPUS_PER_ENV = 8` in `src/poker_ai/cli/train.py` | `_data_parallel_kwargs_for_device` clamps `device_ids` to `range(min(count, 8))` even if 16 NPUs are visible |
| Configs | `poc_asi1_npu.yaml`, `poc_asi2_npu.yaml`, `poc_ai3_npu.yaml` | Policy documented in headers |

- Also patched `scripts/ai3_launch_run7_opponent_mix.sh` (it launches
  `--npus` training directly and previously had no NPU visibility limit).
- Fixed a latent bug: `ASI1_NUM_NPUS`/`ASI2_NUM_NPUS` were documented in the
  train scripts but never consumed — they are now functional.
- Tests: `tests/cli/test_train_devices.py` gains cap-at-8 and
  cap-boundary (exactly 8) cases; all 15 device tests pass.

## Verification

- Full suite (venv Python 3.11): green.
- `bash -n` on all modified scripts; runtime matrix: NUM_NPUS ∈ {1,4,8} →
  correct `ASCEND_RT_VISIBLE_DEVICES`; 9 → exit 1 on ASI1/2/3.
- New exploitability numbers spot-checked against `games/kuhn.py` exact
  evaluator (agreement to 1e-12).

## Notes

- The 8-NPU cap is a *hard* cap: `npu_visible_devices` errors rather than
  silently clamping, so users notice misconfiguration.  `npu_clamped_count`
  (also provided) clamps instead, for future callers that prefer that.
- `ASCEND_RT_VISIBLE_DEVICES` is the standard Ascend torch_npu visibility
  variable; setting it before `run_training.py` starts guarantees even
  single-NPU runs cannot drift onto other chips.
