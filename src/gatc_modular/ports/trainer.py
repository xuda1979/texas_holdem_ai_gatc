from __future__ import annotations

from typing import Any, Mapping, Protocol

from .engine import Engine


class Trainer(Protocol):
    """A generic training interface to decouple training code from engines/policies."""

    def train(self, engine: Engine, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def evaluate(self, engine: Engine, **kwargs: Any) -> Mapping[str, Any]:
        ...

