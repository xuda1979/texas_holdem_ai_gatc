"""Infrastructure for assembling high level subsystems.

The goal of this module is to provide light-weight containers that wrap the
existing implementation units so that they can be reasoned about and tested in
isolation.  Each subsystem exposes a ``component`` attribute pointing at the
underlying implementation (trainer, encoder, etc.) while also recording the
upstream dependencies that were injected during construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, MutableMapping, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class Subsystem(Generic[T]):
    """Wrap a concrete implementation and track its name and dependencies."""

    name: str
    component: T
    dependencies: MutableMapping[str, Any] = field(default_factory=dict)

    def describe(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the subsystem."""

        dependency_names = sorted(self.dependencies)
        component_type = type(self.component).__name__
        return {
            "name": self.name,
            "component": component_type,
            "dependencies": dependency_names,
        }

    def inject(self, key: str, value: Any) -> None:
        """Register an additional dependency after construction."""

        self.dependencies[key] = value

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        return getattr(self.component, item)


@dataclass(slots=True)
class Registry:
    """Keep track of subsystem instances."""

    _items: MutableMapping[str, Subsystem[Any]] = field(default_factory=dict)

    def register(self, subsystem: Subsystem[Any]) -> None:
        if subsystem.name in self._items:
            raise ValueError(f"Subsystem {subsystem.name!r} already registered")
        self._items[subsystem.name] = subsystem

    def get(self, name: str) -> Subsystem[Any]:
        return self._items[name]

    def summary(self) -> list[Mapping[str, Any]]:
        return [subsystem.describe() for subsystem in self._items.values()]
