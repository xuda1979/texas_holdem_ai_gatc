import os
import subprocess
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")


def test_cli_play_runs_banner() -> None:
    cmd = [
        sys.executable,
        "-m",
        "poker_ai.cli.play",
        "--total-players",
        "2",
        "--num-humans",
        "0",
        "--num-hands",
        "1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [env.get("PYTHONPATH"), project_root, src_path])
    )
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "Welcome to Texas Hold'em Poker Simulation" in result.stdout
    assert "Hand Summary:" in result.stdout
