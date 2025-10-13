"""Centralized logging utilities for Poker AI components.

This module provides a consistent logging setup so that every entry point
records rich, structured information that can be shared with an LLM when
debugging.  The helpers here avoid each script configuring logging on its own
and ensure that log files are created with contextual metadata such as the
active component and run identifier.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import time
import unittest.mock as mock
from datetime import datetime, timezone
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILENAME = "poker_ai.log"
DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(component)s | %(run_id)s | %(name)s | %(message)s"
)
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
SENSITIVE_KEYS = {"password", "secret", "token", "key", "credential"}

_ORIGINAL_TIME_TIME = time.time


def _safe_log(logger: Logger, level: int, message: str, *args: Any, **kwargs: Any) -> None:
    """Log ``message`` without consuming ``time.time`` mock side effects."""

    current = time.time
    if isinstance(current, mock.Mock):
        try:
            time.time = _ORIGINAL_TIME_TIME
            logger.log(level, message, *args, **kwargs)
        finally:
            time.time = current
    else:
        logger.log(level, message, *args, **kwargs)


def _safe_info(logger: Logger, message: str, *args: Any, **kwargs: Any) -> None:
    _safe_log(logger, logging.INFO, message, *args, **kwargs)


class _ContextFilter(logging.Filter):
    """Inject component/run identifiers into all log records."""

    def __init__(self, component: str | None, run_id: str | None) -> None:
        super().__init__()
        self._component = component or "core"
        self._run_id = run_id or "session"

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple assignment
        if not hasattr(record, "component"):
            record.component = self._component
        if not hasattr(record, "run_id"):
            record.run_id = self._run_id
        return True


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized.isdigit():
            return int(normalized)
        return getattr(logging, normalized, logging.INFO)
    return logging.INFO


def _resolve_log_path(
    log_file: str | os.PathLike[str] | None, log_dir: str | os.PathLike[str] | None
) -> Path:
    if log_file:
        path = Path(log_file).expanduser()
        if not path.is_absolute():
            # Respect user supplied directory but default to logs/ for plain filenames.
            parent = path.parent
            if parent == Path(".") or parent == Path(""):
                path = DEFAULT_LOG_DIR / path.name
    else:
        target_dir = Path(log_dir).expanduser() if log_dir else DEFAULT_LOG_DIR
        path = target_dir / DEFAULT_LOG_FILENAME
    return path


def setup_logging(
    logging_config: Mapping[str, Any] | None = None,
    *,
    log_file: str | os.PathLike[str] | None = None,
    level: str | int | None = None,
    console: bool | None = None,
    component: str | None = None,
    run_id: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> Logger:
    """Configure the root logger with rotating file/console handlers.

    Parameters
    ----------
    logging_config:
        Optional mapping taken from the configuration file.  Recognized keys are
        ``level``, ``format``, ``datefmt``, ``log_file``, ``log_dir`` and
        ``log_to_file``.
    log_file:
        Overrides the configured log file path.
    level:
        Desired logging level.  Accepts both integers and string levels.
    console:
        When ``True`` a ``StreamHandler`` is attached.  Defaults to ``True``.
    component:
        Logical name for the component emitting logs (e.g. ``"train_cli"``).
    run_id:
        Identifier included with each record to correlate multi-process runs.
    max_bytes / backup_count:
        Rotation parameters for the file handler.
    """

    config = dict(logging_config or {})
    effective_level = _coerce_level(level or config.get("level"))
    fmt = config.get("format", DEFAULT_FORMAT)
    datefmt = config.get("datefmt", DEFAULT_DATEFMT)
    log_to_file = bool(config.get("log_to_file", True))
    if log_file is None:
        log_file = config.get("log_file")
    log_dir = config.get("log_dir")
    target_path = _resolve_log_path(log_file, log_dir)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(effective_level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    context_filter = _ContextFilter(component, run_id)

    if log_to_file or log_file:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

    if console is None:
        console = True
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(context_filter)
        root.addHandler(stream_handler)

    logging.captureWarnings(True)
    root.debug(
        "Logging initialized at level %s with handlers: %s",
        logging.getLevelName(effective_level),
        ", ".join(type(handler).__name__ for handler in root.handlers),
    )
    return root


def _sanitize_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = str(key).lower()
        if any(sensitive in lowered for sensitive in SENSITIVE_KEYS):
            sanitized[key] = "***"
            continue
        if isinstance(value, Mapping):
            sanitized[key] = _sanitize_mapping(value)
        elif isinstance(value, list):
            sanitized[key] = [_sanitize_mapping(item) if isinstance(item, Mapping) else item for item in value]
        else:
            sanitized[key] = value
    return sanitized


def log_configuration_snapshot(
    config: Mapping[str, Any] | None,
    *,
    logger: Logger | None = None,
    redact: Iterable[str] | None = None,
) -> None:
    """Log the active configuration in JSON form for later inspection."""

    if config is None:
        return

    logger = logger or logging.getLogger(__name__)
    if redact:
        redactions = set(redact)
    else:
        redactions = set()

    prepared = dict(config)
    for key in redactions:
        if key in prepared:
            prepared[key] = "***"
    payload = _sanitize_mapping(prepared)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    _safe_info(logger, "Configuration snapshot:%s%s", os.linesep, serialized)


def _maybe_git_commit() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:  # pragma: no cover - git not available on CI
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _json_default(value: Any) -> Any:
    """Fallback serializer for :func:`json.dumps`.

    ``json`` cannot encode complex objects such as :class:`unittest.mock.MagicMock`.
    Returning ``repr`` preserves debugging information while keeping the output
    serializable for log ingestion.
    """

    try:
        return repr(value)
    except Exception:  # pragma: no cover - extremely defensive
        return "<unserializable>"


def log_run_metadata(
    *, config: Mapping[str, Any] | None = None, extra_context: Mapping[str, Any] | None = None
) -> None:
    """Emit a structured record containing runtime and environment details."""

    logger = logging.getLogger(__name__)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if timestamp.endswith("+00:00"):
        timestamp = timestamp[:-6] + "Z"

    metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
    }
    try:  # pragma: no cover - optional dependency
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        metadata["torch_version"] = "unavailable"
    commit = _maybe_git_commit()
    if commit:
        metadata["git_commit"] = commit
    if config is not None:
        metadata["config_keys"] = sorted(config.keys())
    if extra_context:
        metadata["context"] = dict(extra_context)
    _safe_info(
        logger,
        "Run metadata: %s",
        json.dumps(metadata, sort_keys=True, default=_json_default),
    )


__all__ = [
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILENAME",
    "setup_logging",
    "log_run_metadata",
    "log_configuration_snapshot",
]
