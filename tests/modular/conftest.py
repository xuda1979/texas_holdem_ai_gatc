import os
import sys

# Ensure 'src' is on sys.path for src/ layout projects
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Silence pyflakes for unused import when tests only import submodules conditionally
__all__ = ["ROOT", "SRC"]
