# Common Workflows

This guide walks through the three most frequently requested workflows in the
Texas Hold'em AI GATC project: training a model, running automated self-play,
and launching a human-vs-AI match. Each section assumes you have already cloned
the repository and installed the dependencies listed in
[`requirements.txt`](../requirements.txt).

## 1. Train a Model

1. **Activate your environment (optional but recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Start a baseline training run.** This command uses the default Deep CFR
   configuration and writes checkpoints to `trained_models/`.
   ```bash
   python run_training.py
   ```
3. **Customise the run if needed.** For example, to train for 2,000 hands with
   GPU acceleration, call the CLI entry point directly:
   ```bash
   python -m poker_ai.cli.train --num-hands 2000 --gpus
   ```
   Additional options (algorithm, replay buffer sizes, checkpoint frequency,
   etc.) are available via `python -m poker_ai.cli.train -h`.
4. **Locate the checkpoint.** Successful runs create
   `trained_models/cfr_model.pth` (latest checkpoint) and timestamped backups in
   the same directory. These files are consumed by both the self-play pipeline
   and the human-vs-AI interfaces.

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
