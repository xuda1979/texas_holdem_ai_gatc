# Texas Hold'em AI GATC Project

## Overview

This project implements a Texas Hold'em AI using Counterfactual Regret Minimization (CFR) and neural networks built with **PyTorch**. It includes Transformer-based strategies and supports human vs. AI gameplay through a graphical user interface (GUI).

## Project Structure

- **ai_models**: Contains the neural network models (Transformer).
- **trainers**: Includes the AI trainers, self-play scripts, and performance profiling tools.
- **play**: Handles the human vs. AI gameplay and GUI.
- **rules**: Defines the Texas Hold'em rules and game logic. If the optional
  `treys` library is installed, hand evaluation uses it for accurate ranking.
- **tests**: Unit tests for all major components.
- **scripts**: Main scripts for running training and simulations.
- **self_play_data**: Directory for storing self-play results.
- **models**: Stores trained AI models.
- **config.yaml**: Centralized configuration file.


## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training**:
   ```bash
   python run_training.py
   ```

### Training CLI Options

`run_training.py` exposes several command-line flags. Use `-h` to see all options.

Example:

```bash
python run_training.py --num-hands 500 --algorithm deep_cfr --save-model-every 50
```

During training each hand uses a random number of players (between 2 and 10).

3. **Run tests**:
   ```bash
   pytest -q
   ```

4. **Play against the AI**:
   ```bash
   python play/gui.py
   ```

5. **Command-line play simulation**:
   ```bash
   python human_vs_ai.py
   ```

## Features

- Transformer-based neural networks for strategy and advantage estimation
- PyTorch CNN model powering CFR training
- Deep CFR and Single Network CFR trainers
- CFR+ with pruning support
- Opponent modeling with Transformers
- Distributed self-play for faster data collection
- Curriculum learning for staged training
- Attention-based state representation with masks
