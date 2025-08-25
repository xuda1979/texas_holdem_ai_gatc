"""Entry point for launching self-play without requiring package installation."""

import os
import sys

# Allow running from the repository root by ensuring the `src` directory is on
# ``sys.path`` so that the ``poker_ai`` package can be imported without a prior
# installation step.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from poker_ai.cli.self_play import main

if __name__ == "__main__":
    main()
