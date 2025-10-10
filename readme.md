# Texas Hold'em AI GATC Project

[![CI](https://github.com/OWNER/texas_holdem_ai_gatc/actions/workflows/build-and-test.yml/badge.svg?branch=main)](https://github.com/OWNER/texas_holdem_ai_gatc/actions/workflows/build-and-test.yml)

## Table of Contents
- [Introduction](#introduction)
- [Key Capabilities](#key-capabilities)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start Commands](#quick-start-commands)
- [Repository Layout](#repository-layout)
- [Training Pipelines](#training-pipelines)
  - [Core Training CLI](#core-training-cli)
  - [Device Selection (CPU/GPU/NPU/TPU)](#device-selection-cpugpunputpu)
  - [Distributed Self-Play](#distributed-self-play)
  - [Saving and Loading Models](#saving-and-loading-models)
- [Evaluation and Analysis](#evaluation-and-analysis)
  - [Exploitability Utilities](#exploitability-utilities)
  - [Decode and Evaluation Reports](#decode-and-evaluation-reports)
- [Gameplay Options](#gameplay-options)
  - [Command-Line Play](#command-line-play)
  - [Graphical User Interface](#graphical-user-interface)
- [Cloud and Remote Training](#cloud-and-remote-training)
  - [Google Cloud Helper Workflow](#google-cloud-helper-workflow)
- [Testing and Quality Assurance](#testing-and-quality-assurance)
- [Development Workflow](#development-workflow)
  - [Configuration Files](#configuration-files)
  - [Running the Linters and Type Checks](#running-the-linters-and-type-checks)
  - [Working with Examples](#working-with-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository hosts **Texas Hold'em AI GATC**, an end-to-end research platform for
poker strategy development. The project combines classical game-theory
approaches—Counterfactual Regret Minimization (CFR, CFR+, Deep CFR)—with modern
Transformer-based neural networks written in **PyTorch**. It is designed to be a
self-contained playground for experimentation, complete with tooling for
self-play, exploitability analysis, GUI-based human vs. AI matches, and cloud
training workflows.

## Key Capabilities

- Transformer-based neural networks for strategy and advantage estimation.
- Deep CFR, Single Network CFR, CFR+ (with pruning), and discounted CFR+
  solvers.
- Curriculum learning for staged policy refinement and opponent modelling.
- Distributed self-play workers for rapid data collection.
- Optional GPU/NPU/TPU acceleration with minimal configuration.
- Attention-driven state representation and betting-tree abstraction tools.
- Command-line and GUI interfaces for human vs. AI play.
- Utilities for exploitability evaluation, replay buffer inspection, and
  checkpoint management.

## Getting Started

### Prerequisites

- Python 3.10+
- `pip` or `uv` for dependency management
- (Optional) CUDA 11.8 compatible GPU drivers, NPUs, or access to a Cloud TPU VM

### Installation

Clone the repository and install the Python dependencies:

```bash
pip install -r requirements.txt
```

We recommend using a virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`).

### Quick Start Commands

Run a short training session:

```bash
python run_training.py
```

Launch the command-line poker client:

```bash
python -m poker_ai.cli.play
```

Execute the full automated test suite:

```bash
pytest -q
```

Generate the decoder + evaluation report (Chinese documentation):

```bash
python tools/generate_decode_eval_report.py
```

## Repository Layout

All production code lives under `src/poker_ai`, organised into modular
sub-packages:

- `engine/` – core poker engine, player state management, betting logic.
- `rules/` – Texas Hold'em rules, hand evaluation, and CFR helpers.
- `ai/models/` – Transformer, CNN, and advantage network architectures.
- `ai/trainers/` – Deep CFR, Single Network CFR, CFR+ training routines.
- `selfplay/` – multiprocessing self-play workers and replay buffers.
- `evaluation/` – exploitability metrics, benchmarking utilities.
- `gui/` – PyQt/Qt-based human vs. AI interface components.
- `gatc_poker/` – side pot resolution and GATC-specific rule wrappers.
- `cli/` – command-line entry points (`train.py`, `play.py`, `self_play.py`, `gcp_train.py`).
- `config/` – configuration module (`config.py`, `config.yaml`).
- `utils/` – shared helpers, logging utilities, checkpoint storage.

Ancillary tooling lives at the repository root:

- `tools/` – scripts for accelerator provisioning, profiling, and reporting.
- `tests/` – comprehensive unit, integration, and GUI validation suites.
- `examples/` – runnable snippets demonstrating API usage.
- `trained_models/` – created automatically when saving checkpoints.

## Training Pipelines

### Core Training CLI

The primary entry point is `poker_ai.cli.train`. View the available options with:

```bash
python -m poker_ai.cli.train -h
```

Typical usage:

```bash
python -m poker_ai.cli.train --num-hands 500 --algorithm deep_cfr --save-model-every 50
```

The `run_training.py` wrapper ensures that the `poker_ai` package can be imported
without installing the project system-wide. Use it for local experiments or
switch to the module invocation after installing the package.

### Interactive Colab Notebook

An interactive training workflow is available in [`train.ipynb`](train.ipynb).
The notebook is designed for Google Colab and automates the full setup:

1. **Open the notebook.** Either launch it locally with Jupyter or click the
   "Open in Colab" badge at the top of the notebook to run it in Colab.
2. **Configure the run.** Adjust the constants in the first cell (e.g.
   `ALGORITHM`, `NUM_HANDS`, `SAVE_EVERY`, or any extra CLI flags placed in
   `EXTRA_TRAIN_ARGS`). Update the `BRANCH` if you need to test a different git
   branch, and set the optional Google Cloud Storage variables if you want to
   mirror checkpoints.
3. **Execute the main cell.** Running the notebook mounts Google Drive for
   persistent storage, clones or updates the repository, installs dependencies
   from `requirements.txt`, and then starts `poker_ai.cli.train`. Training logs
   and model checkpoints are symlinked to your Drive (`trained_models/`,
   `logs/`, and `reports/`) so they survive across Colab sessions.
4. **Monitor and resume.** Output is streamed live in the notebook while a copy
   is written to Drive. On subsequent runs the notebook automatically resumes
   from the most recent checkpoint discovered in Drive.

The workflow detects GPU availability in Colab and enables `--gpus` when
possible. When TPU support is required, install the `torch-xla` dependency (it
is treated as optional so failures will not abort the setup).

### Device Selection (CPU/GPU/NPU/TPU)

The trainer auto-detects the requested accelerator based on command-line flags:

- `--gpus` – enables PyTorch `DataParallel` across available GPUs.
- `--npus` – mirrors the GPU workflow for NPU devices.
- `--tpu` – configures the training loop for Cloud TPU VMs via `torch_xla`.

For faster iteration lower the replay buffer warm-up with
`--min-buffer-before-train 64`. When not specified, training defaults to the CPU.

#### TPU Workflow Overview

1. Provision a TPU VM that matches the PyTorch/XLA wheel versions.
2. Install the required wheels:
   ```bash
   pip install torch==2.2.0 torch-xla==2.2.0 torchvision==0.17.0 -f \
     https://storage.googleapis.com/tpu-pytorch/wheels/colab.html
   ```
3. Clone this repository, install dependencies, and run:
   ```bash
   python -m poker_ai.cli.train --algorithm deep_cfr --tpu
   ```

Tournament evaluations fall back to the CPU so checkpoints remain compatible
outside of TPU environments.

### Distributed Self-Play

Enable multiprocessing data generation by setting
`training.distributed_workers` in `config.yaml` and then running:

```bash
python -m poker_ai.cli.train --num-hands 1000
```

### Saving and Loading Models

The command-line and GUI clients look for a checkpoint at
`trained_models/cfr_model.pth`. The path is created automatically when using the
training scripts. Custom checkpoints can be loaded via the
`--load-model-path` flag.

## Evaluation and Analysis

### Exploitability Utilities

The `poker_ai.evaluation` package includes helpers for computing exploitability
against simplified matrices:

```python
from poker_ai.evaluation.exploitability import calculate_exploitability
strategy = [0.5, 0.5]
matrix = [[1, -1], [-1, 1]]
print(calculate_exploitability(strategy, matrix))
```

### Decode and Evaluation Reports

To produce a Markdown report summarising decoder metrics, run:

```bash
python tools/generate_decode_eval_report.py
```

The output is written to `reports/decode_eval_report.md` and is useful when
reviewing local changes that affect state decoding or evaluation logic.

## Gameplay Options

### Command-Line Play

Simulate human vs. AI play directly from the terminal:

```bash
python -m poker_ai.cli.play --total-players 2 --num-humans 1 --starting-stack 1000
```

Omitting arguments triggers an interactive prompt for player counts and stack
sizes. For AI-only simulations, launch the self-play harness:

```bash
python self_play.py
```

The project also exposes console entry points when installed via `setup.py`:

```bash
play-poker --total-players 2 --num-humans 1
train-poker --num-hands 500
poker-ai-gcp-train --help
```

### Graphical User Interface

Run the GUI client to challenge the trained agent:

```bash
python app.py
```

Ensure that `trained_models/cfr_model.pth` exists; otherwise, start with the
training pipeline above. The GUI includes seat selection, chip denominations,
card visualization, and betting controls built on the assets in `poker_ai/gui`.

## Cloud and Remote Training

### Google Cloud Helper Workflow

`tools/google_accelerator_train.py` provides a thin wrapper around the `gcloud`
CLI for provisioning GPU or TPU workers and orchestrating remote training. The
three primary phases are:

1. **Create infrastructure**
   - **GPU VM**
     ```bash
     poker-ai-gcp-train \
       --project my-project \
       --zone us-central1-a \
       --name poker-ai-gpu \
       --accelerator gpu \
       --machine-type n1-standard-8 \
       create --gpu-type nvidia-tesla-t4
     ```
   - **TPU VM**
     ```bash
     poker-ai-gcp-train \
       --project my-project \
       --zone us-central1-f \
       --name poker-ai-tpu \
       --accelerator tpu \
       create --tpu-type v4-8 --tpu-version tpu-vm-base
     ```

2. **Run training remotely**
   ```bash
   poker-ai-gcp-train \
     --project my-project \
     --zone us-central1-a \
     --name poker-ai-gpu \
     --accelerator gpu \
     --bucket my-checkpoint-bucket \
     run "python tools/google_accelerator_train.py --accelerator gpu --install-deps --num-hands 2000"
   ```

   The helper synchronises the repository to `~/poker-ai`, optionally installs
   accelerator-specific PyTorch wheels, exports `CHECKPOINT_BUCKET`, and appends
   the appropriate `--gpus`/`--tpu` flag before running the training command. Use
   `--dry-run` to preview commands, `--extra-pip-args` to control wheel sources,
   and `--train-args "--min-buffer-before-train 128"` to forward extra
   parameters.

3. **Clean up resources**
   ```bash
   poker-ai-gcp-train --project my-project --zone us-central1-a --name poker-ai-gpu delete
   ```

Always delete remote infrastructure after use to avoid cloud charges.

## Testing and Quality Assurance

Run the full test suite with:

```bash
pytest -q
```

The suite covers:

- Action validation, deck shuffling, and hand evaluation.
- Engine state transitions and terminal node detection.
- CLI defaults, configuration loading, and argument parsing.
- GUI startup, layout, and bundled asset validation (`tests/gui/test_card_assets.py`).

Smoke tests for the GUI are available via `validate_gui.py` and
`verify_gui_integration.py`.

## Development Workflow

### Configuration Files

Runtime defaults live in `src/poker_ai/config/config.yaml`. Override them via
CLI flags or custom YAML files with the `--config` parameter. The companion
`config.py` module exposes structured accessors for runtime code.

### Running the Linters and Type Checks

This project ships with example scripts for quick verification:

```bash
python example_test_gui_startup.py
python example_test_selfplay.py
```

Adapt them to integrate linting or static analysis tools as needed.

### Working with Examples

The `examples/` directory contains standalone scripts illustrating API usage.
Notable entries include:

- `action_mapping_demo.py` – demonstrates converting discrete action indices to
  betting actions.
- `comprehensive.py` – spins up an end-to-end match with logging enabled.
- `quick.py` – a minimal environment harness for experimentation.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError: poker_ai` | Ensure you are executing commands from the repository root or install the package in editable mode (`pip install -e .`). |
| CUDA devices not detected | Confirm the correct CUDA runtime is installed and pass `--gpus` to the training CLI. Use `nvidia-smi` to verify driver versions. |
| TPU training fails with wheel mismatch | Reinstall `torch`, `torchvision`, and `torch-xla` using versions aligned with the TPU VM image as shown above. |
| GUI assets missing | Run `python setup_card_images.py` or `python tools/verify_images.py` to download and validate card art. |
| Replay buffer fills slowly | Lower `--min-buffer-before-train` or reduce the number of players in the CLI arguments for quicker iterations. |

## Contributing

Issues and pull requests are welcome! Before opening a PR, run the test suite,
format your code, and document user-facing changes. Please include reproduction
steps when reporting bugs to help us diagnose problems quickly.

## License

This project is released under the MIT License. See `LICENSE` for details.
