"""Entry point for launching training without requiring package installation."""

import os
import sys

# Ensure the `src` directory is on the Python path so that the `poker_ai` package
# can be imported when this script is executed directly from the repository
# root.  This avoids ``ModuleNotFoundError`` when the project hasn't been
# installed as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from poker_ai.cli.train import main

if __name__ == "__main__":
    main()
