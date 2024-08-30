# Texas Hold'em AI GATC Project

## Overview

This project implements a Texas Hold'em AI using Counterfactual Regret Minimization (CFR) and neural networks, including Transformer models. The project is designed for Texas Hold'em and supports human vs. AI gameplay with a graphical user interface (GUI).

## Project Structure

- **ai_models**: Contains the neural network models (Transformer).
- **trainers**: Includes the AI trainers, self-play scripts, and performance profiling tools.
- **play**: Handles the human vs. AI gameplay and GUI.
- **rules**: Defines the Texas Hold'em rules and game logic.
- **tests**: Unit tests for all major components.
- **scripts**: Main scripts for running training and simulations.
- **self_play_data**: Directory for storing self-play results.
- **models**: Stores trained AI models.
- **config.yaml**: Centralized configuration file.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
