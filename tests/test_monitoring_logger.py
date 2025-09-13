import json
import os
import tempfile

from poker_ai.monitoring import MetricsLogger, has_nan_or_inf, check_probability_vector


def test_metrics_logger_writes_files():
    with tempfile.TemporaryDirectory() as d:
        logger = MetricsLogger(d)
        logger.log("loss", 1.23, step=5)
        logger.log_many({"a": 0.1, "b": 0.9}, step=6)

        assert os.path.exists(logger.path_jsonl())
        assert os.path.exists(logger.path_csv())

        with open(logger.path_jsonl(), "r", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f.readlines()]
        assert any(l["metric"] == "loss" for l in lines)
        assert any(l.get("step") == 6 for l in lines)


def test_health_checks():
    assert has_nan_or_inf(float("inf"))
    assert has_nan_or_inf(float("nan"))
    assert not has_nan_or_inf(0.0)
    assert check_probability_vector([0.5, 0.5]) is None
    assert check_probability_vector([0.6, 0.6]) is not None

