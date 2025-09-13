import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")


def test_cli_train_runs_without_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    cmd = [
        sys.executable,
        "-m",
        "poker_ai.cli.train",
        "--num-hands",
        "0",
        "--config",
        str(missing),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [env.get("PYTHONPATH"), project_root, src_path])
    )
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "Using default configurations" in result.stdout


def test_cli_train_env_override(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("training:\n  num_training_hands: 100\n")
    cmd = [
        sys.executable,
        "-m",
        "poker_ai.cli.train",
        "--config",
        str(cfg_file),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [env.get("PYTHONPATH"), project_root, src_path])
    )
    env["POKER_AI__TRAINING__NUM_TRAINING_HANDS"] = "0"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "Total training iterations: 0" in result.stdout
