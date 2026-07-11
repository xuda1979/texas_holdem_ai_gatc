"""Entry point for launching comprehensive evaluation without requiring package installation.

Usage
-----

    python evaluate.py checkpoint --path runs/local/models/deep_cfr_final.pth
    python evaluate.py sweep --dir runs/local/models --latest-only
    python evaluate.py health --path runs/local/models/deep_cfr_final.pth
    python evaluate.py h2h --model-a a.pth --model-b b.pth
    python evaluate.py baselines --path runs/local/models/deep_cfr_final.pth

This script adds the ``src`` directory to ``sys.path`` so that the
``poker_ai`` package can be imported when the project hasn't been installed
as a package (e.g. during smoke tests on a remote NPU host).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from poker_ai.cli.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
