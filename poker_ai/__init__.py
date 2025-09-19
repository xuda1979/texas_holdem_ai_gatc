"""Repository-local convenience package for :mod:`poker_ai`.

This shim ensures that the "src" layout works for direct execution without
installing the project.  When running scripts such as ``python -m
poker_ai.cli.play`` from a fresh checkout, Python does not automatically know
about the ``src/`` directory.  By extending the package search path here we make
those modules discoverable while keeping the actual implementation inside
``src/poker_ai``.
"""

from __future__ import annotations

from pkgutil import extend_path
import pathlib
import sys

__all__: list[str] = []

# ``extend_path`` turns this module into a namespace package so that both the
# shim directory and the real implementation under ``src/poker_ai`` contribute
# modules.  This mirrors what ``pip install`` would do but works without
# installation.
__path__ = extend_path(__path__, __name__)

_repo_root = pathlib.Path(__file__).resolve().parent.parent
_src_root = _repo_root / "src"
if _src_root.exists():
    src_entry = str(_src_root)
    if src_entry not in sys.path:
        sys.path.insert(0, src_entry)

    _src_package = _src_root / __name__
    if _src_package.exists():
        # ``extend_path`` may return an object that is not a plain list (it
        # implements the sequence protocol instead).  Convert to a list before
        # mutating so we can safely append the additional search location exactly
        # once.
        paths: list[str] = list(__path__)  # type: ignore[arg-type]
        package_entry = str(_src_package)
        if package_entry not in paths:
            paths.append(package_entry)
        __path__ = paths  # type: ignore[assignment]
