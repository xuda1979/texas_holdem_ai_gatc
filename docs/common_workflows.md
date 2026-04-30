# Common Workflows

This guide walks through the three most frequently requested workflows in the
Texas Hold'em AI GATC project: training a model, running automated self-play,
and launching a human-vs-AI match. Each section assumes you have already cloned
the repository and installed the dependencies listed in
[`requirements.txt`](../requirements.txt).

## 1. Train a Model

This workspace's default training target is the Huanxin `ai1` environment. Keep
large checkpoints and model artifacts on `ai1` (or S3), not in the local
workspace.

1. **Run fast local validation first.** Use targeted tests before each remote
   training push so broken code never reaches `ai1`.
   ```bash
   python3.11 -m unittest tests.test_train_cli_observability_unittest -v
   python3.11 -m pytest -q tests/cli/test_train_devices.py tests/test_self_play.py
   ```
2. **Sync the current workspace to S3.**
   ```bash
   scripts/push_to_s3.sh
   ```
3. **Sync the remote ai1 worktree from S3.**
   ```bash
   scripts/ai1_sync_from_s3.sh
   ```
4. **Launch training on ai1.** Use the job wrapper for long runs so logs remain
   attached to a named remote process.
   ```bash
   scripts/ai1_job.sh start deep-cfr-train /tmp/ai1-deep-cfr-train.log \
     "cd /root/root/work/texas-holdem && \
      python3.11 -m poker_ai.cli.train \
        --algorithm deep_cfr \
        --npus \
        --num-hands 200000 \
        --samples-per-cycle 512 \
        --train-steps-per-cycle 32 \
        --save-model-every 10 \
        --save-samples 5000"
   ```
5. **Monitor the structured training events.** The train CLI now emits stable
   JSON payloads in the log stream using the `Training event:` prefix. Tail the
   remote log and filter for those machine-readable markers.
   ```bash
   scripts/ai1_shell.sh "grep 'Training event:' /tmp/ai1-deep-cfr-train.log | tail -n 20"
   ```
   The most important events are:
   - `checkpoint_saved` — periodic hand/time checkpoint persisted on ai1
   - `cycle_complete` — a generate/train cycle finished, including sample count,
     replay buffer size, step count, and average loss when available
   - `evaluation_status` — post-cycle analyzer decision, including
     `no_improvement_samples` and whether training should continue
   - `final_model_saved` — final artifact path after normal completion or early
     stop
6. **Export results from ai1 after the run.**
   ```bash
   scripts/ai1_push_results_to_s3.sh outputs models
   ```
7. **Pull back only small diagnostics if needed.** Keep large checkpoints on
   ai1/S3. Review logs, summaries, and evaluation outputs locally; do not keep
   full model artifacts in this repository.

## 2. Run Self-Play Simulations

1. **Ensure a checkpoint exists.** Use the latest model saved in
   `trained_models/` or allow the script to initialise a new model when none is
   present.
2. **Launch the self-play harness.**
   ```bash
   python -m poker_ai.cli.self_play
   ```
   The command streams progress to the console while periodically persisting
   updated checkpoints to `trained_models/`. Pass `--config path/to/config.yaml`
   to override defaults such as the number of parallel workers or save
   intervals.
3. **Inspect results.** Summary statistics are emitted to stdout, and the most
   recent model snapshot is stored in `trained_models/` with a timestamped
   filename.

## 3. Play Human vs. AI

You can challenge the latest model either through the command-line client or
the full graphical interface.

### Command-Line Client

```bash
python -m poker_ai.cli.play --total-players 2 --num-humans 1 --starting-stack 1000
```

- When `--num-humans` is omitted the script prompts for player counts
  interactively.
- Use `--load-model-path` to point at a specific checkpoint if you do not want
  the default `trained_models/cfr_model.pth` file.

### Graphical Interface

```bash
python app.py
```

- The GUI automatically loads `trained_models/cfr_model.pth` if it exists. If
  the file is missing, run the training workflow above to produce one.
- Seat selection, chip denominations, and betting controls are available inside
  the interface once it launches.

## Tips

- **Version control:** Check your working tree before launching long training
  jobs so that checkpoints and logs do not pollute unrelated branches.
- **Hardware:** GPU runs require CUDA-compatible drivers. For TPU/NPU guidance,
  review the dedicated sections in `readme.md`.
- **Automation:** Combine these commands in shell scripts or notebooks (see
  `train.ipynb`) to reproduce experiments quickly.
