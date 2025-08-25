"""Light‑weight CLI entry points.

The original module imported all subcommands eagerly which pulled in heavy
dependencies (e.g. ``yaml`` from :mod:`poker_ai.cli.train`) even when only the
``self_play`` command was required.  This caused import failures in minimal
environments.  To keep the package importable without optional dependencies we
expose thin wrapper functions that perform the imports lazily.
"""

from typing import Any


def train_main(*args: Any, **kwargs: Any) -> Any:
    from .train import main as _main
    return _main(*args, **kwargs)


def play_main(*args: Any, **kwargs: Any) -> Any:
    from .play import main as _main
    return _main(*args, **kwargs)


def self_play_main(*args: Any, **kwargs: Any) -> Any:
    from .self_play import main as _main
    return _main(*args, **kwargs)


__all__ = ["train_main", "play_main", "self_play_main"]
