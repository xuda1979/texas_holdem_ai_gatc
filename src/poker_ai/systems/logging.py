"""Subsystem for application logging utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_ai import logging_utils

from .base import Subsystem


@dataclass
class LoggingSubsystem(Subsystem[logging_utils]):
    """Expose logging helpers for other subsystems."""

    @classmethod
    def create(cls, **kwargs: Any) -> "LoggingSubsystem":
        subsystem = cls(name="logging", component=logging_utils)
        if kwargs:
            subsystem.dependencies.update(kwargs)
        return subsystem

    def setup(self, *args: Any, **kwargs: Any) -> None:
        logging_utils.setup_logging(*args, **kwargs)

    def snapshot(self, *args: Any, **kwargs: Any) -> None:
        logging_utils.log_configuration_snapshot(*args, **kwargs)

    def run_metadata(self, *args: Any, **kwargs: Any) -> None:
        logging_utils.log_run_metadata(*args, **kwargs)
