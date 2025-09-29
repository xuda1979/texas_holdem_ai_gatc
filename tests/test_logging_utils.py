import logging

from poker_ai.logging_utils import (
    log_configuration_snapshot,
    log_run_metadata,
    setup_logging,
)


def test_setup_logging_writes_file(tmp_path):
    log_path = tmp_path / "app.log"
    setup_logging(
        {"log_file": str(log_path), "level": "INFO", "log_to_file": True},
        component="test",
        run_id="unit",
        console=False,
    )
    logger = logging.getLogger("testcase")
    logger.info("hello world")
    log_configuration_snapshot({"api": {"password": "secret", "timeout": 5}})
    log_run_metadata(config={"foo": "bar"})

    for handler in logging.getLogger().handlers:
        handler.flush()

    contents = log_path.read_text()
    assert "hello world" in contents
    assert "***" in contents  # password redacted
    assert "Run metadata" in contents

    logging.shutdown()
