import importlib
from pathlib import Path

import pytest

logging_utils = importlib.import_module("poker_ai.logging_utils")


@pytest.mark.parametrize(
    "log_file, expected",
    [
        ("custom.log", logging_utils.DEFAULT_LOG_DIR / "custom.log"),
        ("logs/custom.log", Path("logs/custom.log")),
        ("/tmp/custom.log", Path("/tmp/custom.log")),
    ],
)
def test_resolve_log_path_defaults_to_logs_dir(tmp_path, log_file, expected):
    result = logging_utils._resolve_log_path(log_file, None)
    assert result == expected


def test_resolve_log_path_with_log_dir(tmp_path):
    custom_dir = tmp_path / "nested"
    result = logging_utils._resolve_log_path(None, custom_dir)
    assert result == custom_dir / logging_utils.DEFAULT_LOG_FILENAME
