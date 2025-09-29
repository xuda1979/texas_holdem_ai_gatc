# Observability and Diagnostics

This repository now ships with a centralized logging configuration and a small
set of repeatable diagnostics that make it easier to reason about training
runs.  The goal is to capture enough detail that a log file can be handed to an
LLM for post-mortem analysis without additional context.

## Centralized logging

* The helper :func:`poker_ai.logging_utils.setup_logging` configures both
  rotating file and console handlers.  By default logs are written under
  ``logs/poker_ai.log`` with a timestamped, component-aware format.
* ``setup_logging`` accepts the ``logging`` section from
  ``config/config.yaml`` and automatically creates directories as needed.  Each
  record contains the component name and run identifier so that multi-process
  workflows remain distinguishable.
* Call :func:`poker_ai.logging_utils.log_run_metadata` and
  :func:`poker_ai.logging_utils.log_configuration_snapshot` at the start of a
  run to capture environment details (Python/Torch versions, git commit, CLI
  arguments, etc.) and the sanitized configuration that produced the run.
* The training CLI, trainers, self-play loop, and evaluation utilities now emit
  structured INFO-level messages whenever models are saved, training steps
  complete, or significant events occur.  Rotating file handlers prevent logs
  from growing without bound.

## Quick diagnostics for algorithm validation

Use the ``tools/run_diagnostics.py`` script to execute a set of fast checks that
exercise the critical pieces of the CFR training pipeline:

```bash
python tools/run_diagnostics.py --algorithm ai_cfr --output diagnostics.json
```

The script performs the following actions (each recorded in the log and
optionally in the emitted JSON report):

1. **Forward pass check** – runs a single inference through the trainer's model
   to ensure tensor shapes are consistent and logs the execution time.
2. **Training step check** – performs a small gradient update to confirm that
   the optimizer, loss computation, and gradients are wired correctly.
3. **Optional self-play check** – include ``--include-self-play`` to execute a
   lightweight MCCFR hand via ``SelfPlay``.  This takes longer but validates the
   integration between the environment and the trainer.

These diagnostics are light-weight enough to run before long training jobs or
after code changes to validate algorithmic correctness and approximate
efficiency.  Combine them with the detailed log files to quickly spot
regressions or feed concise transcripts into an LLM for debugging assistance.
