import os
import subprocess
import sys
from pathlib import Path

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
