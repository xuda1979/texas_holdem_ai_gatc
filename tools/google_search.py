"""Google Search tool interface and deterministic datetime overrides.

This module exposes a thin wrapper over the sandbox RPC interface used by
internal tooling.  It also provides deterministic replacements for
``datetime.date`` and ``datetime.datetime`` so that integrations relying on the
current time behave predictably in hermetic test environments.

The implementation borrows heavily from a reference implementation but adds
compatibility tweaks and additional documentation.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

try:  # pragma: no cover - optional dependency in open-source release
    from google3.assistant.boq.lamda.execution_box.sandbox_interface import sandbox_rpc
except ImportError:  # pragma: no cover - fallback for open-source environment
    from sandbox_interface import sandbox_rpc  # type: ignore[import-not-found]

import datetime


class GoogleSearch:
    """Interface for invoking Google Search through the sandbox RPC layer."""

    _tool_name = "google_search"

    @dataclasses.dataclass
    class PerQueryResult:
        """Represents a single search result returned by the service."""

        index: Optional[str] = None
        publication_time: Optional[str] = None
        snippet: Optional[str] = None
        source_title: Optional[str] = None
        url: Optional[str] = None

    @dataclasses.dataclass
    class SearchResults:
        """Container holding the results for a single query string."""

        query: Optional[str] = None
        results: Optional[List["GoogleSearch.PerQueryResult"]] = None

    def search(
        self,
        queries: Optional[List[str]] = None,
    ) -> List["GoogleSearch.SearchResults"]:
        """Execute Google search queries and return structured results.

        Args:
            queries: An optional list of query strings to execute.

        Returns:
            A list of :class:`SearchResults` objects describing the results for
            each query in ``queries``.
        """

        parameters = {
            "queries": queries,
        }

        return sandbox_rpc.run_tool_parse_result(
            name="google_search",
            operation_id="search",
            parameters=parameters,
            return_type=List[GoogleSearch.SearchResults],
            scope=globals(),
            strict=True,
            ignore_unrecognized_fields=True,
            model_invisible_fields=[],
        )


google_search = GoogleSearch()


class CustomDate(datetime.date):
    """Deterministic stand-in for :class:`datetime.date`."""

    @classmethod
    def today(cls) -> "CustomDate":
        """Return a fixed date for repeatable tests."""

        # Use ``cls`` to respect subclassing and to mirror the original API.
        return cls(year=2025, month=10, day=13)


class CustomDateTime(datetime.datetime, CustomDate):
    """Deterministic ``datetime`` implementation mirroring Python's hierarchy."""

    @classmethod
    def now(cls, tz: Optional[datetime.tzinfo] = None) -> "CustomDateTime":
        """Return a fixed point in time, optionally adjusted for ``tz``."""

        value = cls(year=2025, month=10, day=13, hour=21, minute=52, second=0)
        if tz is not None:
            value = value.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
        return value

    @classmethod
    def today(cls) -> "CustomDateTime":
        """Maintain compatibility with :meth:`datetime.datetime.today`."""

        return cls(year=2025, month=10, day=13, hour=21, minute=52, second=0)


# NOTE: This global monkey-patching strategy is fragile and dependent on import
# order.  Any module that imports ``datetime`` *before* these assignments will
# continue to see the real classes.  The approach works for the controlled
# environments where this module is used but should be replaced with a dedicated
# time-freezing library for more complex applications.
datetime.datetime = CustomDateTime
datetime.date = CustomDate
