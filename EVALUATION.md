# Comprehensive Evaluation System

This document describes the comprehensive model evaluation system for the poker
AI, designed to be used **during training** (via in-loop hooks) and **after
training** (via checkpoint sweeps) to quickly find bugs, regressions, and
training issues.

## Quick start

### Evaluate a single checkpoint

```bash
# Human-readable summary
python evaluate.py checkpoint --path models/deep_cfr_final.pth

# Machine-readable JSON report
python evaluate.py checkpoint --path models/deep_cfr_final.pth \
    --h2h-hands 200 --health-states 32 \
    --output reports/ckpt_report.json
```

### Sweep every checkpoint in a directory

```bash
# Evaluate all checkpoints (newest last)
python evaluate.py sweep --dir models/ --output reports/sweep.json

# Evaluate only the latest checkpoint
python evaluate.py sweep --dir models/ --latest-only --output reports/latest.json

# Track regressions across sweeps
python evaluate.py sweep --dir models/ \
    --tracker-path reports/eval_history.json \
    --regression-margin 10.0
```

### Run only the (fast) health check

```bash
python evaluate.py health --path models/deep_cfr_final.pth
```

### Compare two checkpoints head-to-head

```bash
python evaluate.py h2h \
    --model-a models/deep_cfr_hand_1000.pth \
    --model-b models/deep_cfr_hand_2000.pth \
    --hands 500 --output reports/h2h.json
```

### Play a model against the baseline panel

```bash
python evaluate.py baselines --path models/deep_cfr_final.pth --hands 200
```

## What the evaluation battery measures

### 1. Health checks (`poker_ai.evaluation.health`)

Non-gameplay diagnostics that inspect the network directly. These are **cheap**
and catch the vast majority of training bugs:

- **Weight NaN/Inf**: any NaN/Inf in parameter tensors → error.
- **Weight max-abs**: largest |weight| > 1e4 → error (exploding weights).
- **Dead parameters**: all-zero parameter tensors → warning.
- **Policy NaN/Inf**: NaN/Inf in advantage outputs across sampled states → error.
- **Policy determinism**: advantage outputs must be deterministic in eval mode.
- **Policy argmax legality**: argmax must always fall on a legal action.
- **Policy legal mass**: total probability on legal actions must be 1.0.
- **Policy entropy**: low entropy → collapsed policy warning.
- **Metadata consistency**: hidden_dim, num_actions, etc. match expected values.

### 2. Head-to-head matches (`poker_ai.evaluation.head_to_head`)

Plays duplicate hands (same deck seed, both seats) against a fixed panel of
baselines and reports win rate in **bb/100** (big blinds per 100 hands) with a
standard error. The panel:

- **always_fold**: folds every decision. *Losing to this is always a bug.*
- **calling_station**: checks/calls every decision, never bets/raises/folds.
- **random_uniform**: picks any legal action uniformly at random.

A new checkpoint is flagged as a regression if its bb/100 against a baseline
drops by more than `--regression-margin` (default 10 bb/100) below the best
previously observed value.

### 3. Kuhn-poker exploitability (optional, `--run-kuhn`)

Computes exact exploitability for a Kuhn-poker CFR solution as a tractable
proxy for Nash convergence. Full Texas Hold'em best response is intractable, so
Kuhn poker serves as a sanity check that the CFR algorithm itself is correct.

### 4. Regression detection (`RegressionTracker`)

Persists a rolling history of evaluation reports to disk and flags
regressions:

```python
from poker_ai.evaluation.comprehensive import RegressionTracker

tracker = RegressionTracker(history_path="eval_history.json", margin_bb=10.0)
warnings = tracker.update(report)
for w in warnings:
    report.update_severity("warning", w)
```

## In-training evaluation

The training CLI reads the `evaluation` section of the config and wires up a
`TrainingEvalHook` that runs the full battery periodically during training:

```yaml
# poc_ai3_npu.yaml
evaluation:
  every_samples: 1000       # run the battery every 1000 samples
  h2h_hands: 200            # hands per head-to-head match
  health_num_states: 32     # random states for health check
  seed: 7                   # reproducible evaluation
  emit_events: true         # emit JSON training events
  stop_on_error: false      # set true to abort training on error-severity
```

When `every_samples <= 0` (or the `evaluation` section is absent), the hook is
disabled and the training loop runs unchanged.

The hook emits `__TRAINING_EVENT__ comprehensive_evaluation {...}` JSON lines
to the training log; downstream tooling can grep for these to extract the
evaluation stream.

## ASI3 (Ascend NPU) workflow

### Launch a training run on ASI3

```bash
# Start a training job (syncs the repo first, then launches on ASI3)
scripts/asi3_train.sh start my-run-001

# Check status
scripts/asi3_train.sh status my-run-001-20260703T120000Z-12345

# Tail logs
scripts/asi3_train.sh logs my-run-001-20260703T120000Z-12345

# List all jobs
scripts/asi3_train.sh list
```

### Evaluate checkpoints on ASI3

```bash
# Health-check a checkpoint (runs on ASI3)
scripts/asi3_evaluate.sh health ~/software/texas-holdem/models/deep_cfr_final.pth

# Sweep all checkpoints on ASI3
scripts/asi3_evaluate.sh sweep ~/software/texas-holdem/models --latest-only

# Compare two checkpoints
scripts/asi3_evaluate.sh h2h \
    ~/software/texas-holdem/models/deep_cfr_hand_1000.pth \
    ~/software/texas-holdem/models/deep_cfr_hand_2000.pth
```

### Pull artifacts back to your laptop

```bash
# Pull models + logs + eval history to ./runs/asi3/
scripts/asi3_evaluate.sh pull runs/asi3/

# Then sweep locally for faster iteration
python evaluate.py sweep --dir runs/asi3/models --output runs/asi3/sweep.json
```

## Reading an evaluation report

```json
{
  "checkpoint_path": "models/deep_cfr_final.pth",
  "created_at": "2026-07-03T12:00:00Z",
  "total_samples": 20000,
  "severity": "ok",           // "ok" | "warning" | "error"
  "ok": true,
  "issues": [],               // list of human-readable issue strings
  "health_ok": true,
  "health_severity": "ok",
  "health_checks": {          // detailed per-check results
    "weight_nan_count": {"severity": "ok", "value": 0, "message": "..."},
    "policy_legal_mass": {"severity": "ok", "value": 1.0, "message": "..."},
    ...
  },
  "health_diagnostics": {     // numeric diagnostics for trending
    "weight_total_params": 7124874,
    "policy_entropy_min": 0.0001,
    "policy_entropy_max": 1.38,
    ...
  },
  "h2h_results": [            // one entry per baseline
    {
      "baseline": "always_fold",
      "hands_played": 200,
      "bb_per_100": 75.0,
      "stderr_bb_per_100": 5.0,
      "is_significantly_positive": true,
      "is_significantly_negative": false
    },
    ...
  ],
  "kuhn_exploitability": null,  // or {"exploitability": 0.0, "converged": true, ...}
  "metadata": {                // checkpoint metadata
    "history_feature_dim": 18,
    "hidden_dim": 128,
    ...
  },
  "diagnostics": {             // eval-run diagnostics
    "elapsed_seconds": 1.55
  }
}
```

## Severity escalation

The report severity is the worst of:

- `health_severity` (from health checks)
- `error` if h2h against `always_fold` is significantly negative
- `warning` if h2h against `calling_station` is significantly negative
- `warning` if Kuhn exploitability is high or unconverged
- `warning` for each regression flagged by `RegressionTracker`

`stop_on_error: true` in the config will cause the training loop to abort when
the report severity is `error`.

## Programmatic API

```python
from poker_ai.evaluation.comprehensive import (
    ComprehensiveEvaluator,
    EvaluationReport,
    RegressionTracker,
    TrainingEvalHook,
    discover_checkpoints,
    load_model_from_checkpoint,
)

# Evaluate one checkpoint
evaluator = ComprehensiveEvaluator(h2h_hands=200, health_num_states=32, seed=7)
report = evaluator.evaluate_checkpoint("models/deep_cfr_final.pth", total_samples=20000)
print(report.summary)
print(report.to_json())

# Sweep a directory
for ckpt in discover_checkpoints("models/", prefix="deep_cfr_"):
    report = evaluator.evaluate_checkpoint(str(ckpt))
    # ... save report, update tracker, etc.

# Load a model directly for custom evaluation
model, metadata, device = load_model_from_checkpoint("models/deep_cfr_final.pth")
```

## Troubleshooting

### "Local model/checkpoint writes are disabled"

Set `POKER_AI_ALLOW_LOCAL_MODEL_WRITES=1` for local smoke tests, or
`POKER_AI_HUANXIN_REMOTE_ROOT=/tmp/asi3_smoke` to redirect model writes to a
local directory.

### "Could not parse job PID from transcript"

The ASI3 job launcher couldn't find the PID marker in the remote shell output.
This usually means the remote command failed before backgrounding. Check the
full transcript with `scripts/asi3_job.sh logs <job-id>`.

### Health check reports "policy_collapsed"

The model puts >99.9% probability on a single action across all sampled
states. This is a common failure mode early in training when the advantage
network hasn't seen enough data. If it persists after >1000 samples, check:

- The replay buffer isn't empty (`min_buffer_before_train` is set correctly).
- The learning rate isn't too high (causing the policy to collapse to a
  dominant action).
- The action mask is being applied correctly in `prepare_transformer_input`.

### H2H against `always_fold` is significantly negative

This is always a bug. The model is *folding when it should call* (since
always-fold never bets, calling is free). Check:

- The action mapping in `get_action_from_index` is correct.
- The model's argmax isn't landing on an illegal action (check
  `policy_argmax_illegal` in the health report).
- The `ModelAIStrategy.choose_action` is using the legal mask correctly.
