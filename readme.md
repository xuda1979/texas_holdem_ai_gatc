# Texas Hold'em AI GATC Project

## Overview

This project implements a Texas Hold'em AI using Counterfactual Regret Minimization (CFR) and neural networks built with **PyTorch**. It includes Transformer-based strategies and supports human vs. AI gameplay through a graphical user interface (GUI).

## Project Structure

All production code now lives under `src/poker_ai` to allow modular development.

- `engine/` – core game engine implementation.
- `rules/` – poker rules and CFR utilities.
- `ai/models/` – neural network models.
- `ai/trainers/` – training algorithms and helpers.
- `selfplay/` – parallel self‑play environment.
- `gui/` – human vs. AI interface and GUI utilities.
- `utils/` – supporting utilities.
- `evaluation/` – exploitability and analysis tools.
- `config/` – configuration module containing `config.py` and `config.yaml`.
- `cli/` – command line entry points (`train.py`, `play.py`, `self_play.py`).


## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training**:
   ```bash
   python -m poker_ai.cli.train
   ```

### Training CLI Options

`poker_ai.cli.train` exposes several command-line flags. Use `-h` to see all options.

Example:

```bash
python -m poker_ai.cli.train --num-hands 500 --algorithm deep_cfr --save-model-every 50
```

During training each hand uses a random number of players (between 2 and 10).

3. **Run tests**:
   ```bash
   pytest -q
   ```

4. **Play against the AI**:
   ```bash
   python -m poker_ai.cli.play
   ```

5. **Command-line play simulation**:
   ```bash
   python -m poker_ai.cli.play --total-players 2 --num-humans 1 --starting-stack 1000
   ```
   The script falls back to interactive prompts if arguments are omitted.

   After installation via `setup.py`, you can also use the entry points:
   ```bash
   play-poker --total-players 2 --num-humans 1
   train-poker --num-hands 500
  ```

### Trained Model

`poker_ai.cli.play` and the GUI expect a model file at `trained_models/cfr_model.pth`.
Run the training script to generate it:

```bash
python -m poker_ai.cli.train
```

The directory `trained_models/` will be created automatically when saving.

## Features

- Transformer-based neural networks for strategy and advantage estimation
- PyTorch CNN model powering CFR training
- Deep CFR and Single Network CFR trainers
- CFR+ with pruning support
- Opponent modeling with Transformers
- Distributed self-play for faster data collection
- Curriculum learning for staged training
- Attention-based state representation with masks
- Betting-tree abstraction utilities
- Discounted CFR+ solver with regret discounting
- Distributed self-play with multiprocessing
- Optional GPU acceleration with automatic device selection
- Simple exploitability evaluation tools

## Using Distributed Self-Play

Set `training.distributed_workers` in `config.yaml` to the number of worker
processes. Then run training normally:

```bash
python run_training.py --num-hands 1000
```

## Evaluating Strategies

The `evaluation` package offers a lightweight exploitability calculator. Example:

```python
from evaluation.exploitability import calculate_exploitability
strategy = [0.5, 0.5]
matrix = [[1, -1], [-1, 1]]
print(calculate_exploitability(strategy, matrix))
```

## Examples

Demonstration scripts are provided in the `examples` directory. For instance,
`examples/action_mapping_demo.py` showcases how to use the `get_action_from_index`
helper to map action indices to in-game actions.
