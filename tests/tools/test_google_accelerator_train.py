from __future__ import annotations

from types import SimpleNamespace

from tools import google_accelerator_train as helper


def test_install_gpu_packages(monkeypatch):
    recorded = []
    monkeypatch.setattr(helper, "_run_command", lambda cmd, **kw: recorded.append((cmd, kw)))
    args = SimpleNamespace(
        torch_version="2.1.0",
        torchvision_version="0.16.0",
        torchaudio_version="2.1.0",
        torch_index_url="https://download.pytorch.org/whl/cu121",
        extra_pip_args=["--extra-index-url", "https://example.com/wheels"],
    )
    helper._install_gpu_packages(args, dry_run=False)
    command, kwargs = recorded[0]
    assert command[:4] == [helper.sys.executable, "-m", "pip", "install"]
    assert "--extra-index-url" in command
    assert kwargs == {"dry_run": False}


def test_install_tpu_packages(monkeypatch):
    recorded = []
    monkeypatch.setattr(helper, "_run_command", lambda cmd, **kw: recorded.append((cmd, kw)))
    args = SimpleNamespace(
        torch_version="2.1.0",
        torchvision_version="0.16.0",
        torch_xla_version="2.1.0",
        tpu_wheel_url="https://storage.googleapis.com/tpu-pytorch/wheels.html",
        extra_pip_args=None,
    )
    helper._install_tpu_packages(args, dry_run=True)
    command, kwargs = recorded[0]
    assert "torch-xla==2.1.0" in command
    assert "-f" in command
    assert kwargs == {"dry_run": True}


def test_build_training_command_gpu():
    args = SimpleNamespace(
        accelerator="gpu",
        algorithm="deep_cfr",
        num_hands=100,
        config="config.yaml",
        save_model_every=10,
        train_args="--min-buffer-before-train 64",
    )
    command = helper._build_training_command(args)
    assert "--gpus" in command
    assert command.count("--num-hands") == 1
    assert command[-2:] == ["--min-buffer-before-train", "64"]


def test_build_training_command_tpu():
    args = SimpleNamespace(
        accelerator="tpu",
        algorithm=None,
        num_hands=None,
        config=None,
        save_model_every=None,
        train_args=None,
    )
    command = helper._build_training_command(args)
    assert command[-1] == "--tpu"


def test_run_command_dry_run(monkeypatch):
    recorded = []

    def fake_run(cmd, check=True, env=None):  # pragma: no cover - should not run
        recorded.append((cmd, env))

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    helper._run_command(["echo", "hello"], dry_run=True)
    assert recorded == []


def test_run_command_executes(monkeypatch):
    recorded = []

    def fake_run(cmd, check=True, env=None):
        recorded.append((cmd, env))

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    helper._run_command(["echo", "hello"], dry_run=False)
    assert recorded[0][0] == ["echo", "hello"]


def test_main_installs_and_runs(monkeypatch):
    recorded = []

    def fake_install_base(*, dry_run):
        recorded.append(("base", dry_run))

    def fake_install_accelerator(args, *, dry_run):
        recorded.append((args.accelerator, dry_run))

    def fake_run_command(command, *, env, dry_run):
        recorded.append(("run", command, env, dry_run))

    monkeypatch.setattr(helper, "_install_base_requirements", fake_install_base)
    monkeypatch.setattr(helper, "_install_gpu_packages", fake_install_accelerator)
    monkeypatch.setattr(helper, "_install_tpu_packages", fake_install_accelerator)
    monkeypatch.setattr(helper, "_run_command", fake_run_command)
    monkeypatch.setattr(helper.os, "environ", {"PATH": "/usr/bin"})

    args = SimpleNamespace(
        accelerator="gpu",
        install_deps=True,
        bucket="gs://bucket",
        dry_run=True,
        algorithm="deep_cfr",
        num_hands=None,
        config=None,
        save_model_every=None,
        train_args=None,
    )

    monkeypatch.setattr(helper, "parse_args", lambda _: args)
    helper.main([])

    assert recorded[0] == ("base", True)
    assert recorded[1][0] == "gpu"
    assert recorded[-1][0] == "run"
    assert recorded[-1][2]["CHECKPOINT_BUCKET"] == "gs://bucket"
