from __future__ import annotations

from types import SimpleNamespace

import pytest

from poker_ai.cli import gcp_train


def test_ensure_gcloud_available(monkeypatch):
    monkeypatch.setattr(gcp_train.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as excinfo:
        gcp_train.ensure_gcloud_available()
    assert "gcloud" in str(excinfo.value)


def test_ensure_gcloud_present(monkeypatch):
    monkeypatch.setattr(gcp_train.shutil, "which", lambda _: "/usr/bin/gcloud")
    gcp_train.ensure_gcloud_available()  # Should not raise


def test_run_command_failure(monkeypatch):
    def fake_run(cmd, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    import subprocess

    monkeypatch.setattr(gcp_train.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as excinfo:
        gcp_train._run_command(["gcloud", "version"])
    assert "exit code 1" in str(excinfo.value)


def test_create_instance_gpu(monkeypatch):
    recorded = []
    monkeypatch.setattr(gcp_train, "ensure_gcloud_available", lambda: None)
    monkeypatch.setattr(gcp_train, "_run_command", lambda cmd: recorded.append(cmd))

    args = SimpleNamespace(
        accelerator="gpu",
        gpu_type="nvidia-a100-80gb",
        gpu_count=2,
        image=None,
        image_family="pytorch-latest-gpu",
        image_project="deeplearning-platform-release",
        name="trainer",
        project="proj",
        zone="us-central1-a",
        machine_type="n1-standard-8",
        tpu_type=None,
        tpu_version="tpu-vm-base",
    )

    gcp_train.create_instance(args)

    assert recorded[0][:7] == [
        "gcloud",
        "compute",
        "instances",
        "create",
        "trainer",
        "--project",
        "proj",
    ]
    assert "--accelerator" in recorded[0]
    assert "type=nvidia-a100-80gb,count=2" in recorded[0]


def test_create_instance_tpu(monkeypatch):
    recorded = []
    monkeypatch.setattr(gcp_train, "ensure_gcloud_available", lambda: None)
    monkeypatch.setattr(gcp_train, "_run_command", lambda cmd: recorded.append(cmd))

    args = SimpleNamespace(
        accelerator="tpu",
        gpu_type=None,
        gpu_count=0,
        image=None,
        image_family=None,
        image_project=None,
        name="trainer",
        project="proj",
        zone="us-central1-b",
        machine_type="n1-standard-8",
        tpu_type="v4-8",
        tpu_version="tpu-vm-base",
    )

    gcp_train.create_instance(args)

    assert recorded[0][:5] == ["gcloud", "alpha", "compute", "tpus", "tpu-vm"]
    assert "--accelerator-type" in recorded[0]
    assert "v4-8" in recorded[0]


def test_run_training_gpu(monkeypatch):
    recorded = []
    monkeypatch.setattr(gcp_train, "ensure_gcloud_available", lambda: None)
    monkeypatch.setattr(gcp_train, "_run_command", lambda cmd: recorded.append(cmd))
    monkeypatch.setattr(gcp_train, "_repo_root", lambda: gcp_train.Path("/repo"))

    args = SimpleNamespace(
        accelerator="gpu",
        name="trainer",
        project="proj",
        zone="us-central1-a",
        command="python -m poker_ai.cli.train --gpus",
        bucket=None,
    )

    gcp_train.run_training(args)

    sync_cmd, ssh_cmd = recorded
    assert sync_cmd[:3] == ["gcloud", "compute", "scp"]
    assert ssh_cmd[:3] == ["gcloud", "compute", "ssh"]
    assert "--command" in ssh_cmd
    command_index = ssh_cmd.index("--command") + 1
    assert ssh_cmd[command_index].endswith("python -m poker_ai.cli.train --gpus")


def test_run_training_tpu(monkeypatch):
    recorded = []
    monkeypatch.setattr(gcp_train, "ensure_gcloud_available", lambda: None)
    monkeypatch.setattr(gcp_train, "_run_command", lambda cmd: recorded.append(cmd))
    monkeypatch.setattr(gcp_train, "_repo_root", lambda: gcp_train.Path("/repo"))

    args = SimpleNamespace(
        accelerator="tpu",
        name="trainer",
        project="proj",
        zone="us-central1-b",
        command="python -m poker_ai.cli.train --tpu",
        bucket="gs://bucket",
    )

    gcp_train.run_training(args)

    sync_cmd, ssh_cmd = recorded
    assert sync_cmd[:5] == ["gcloud", "alpha", "compute", "tpus", "tpu-vm"]
    assert ssh_cmd[:5] == ["gcloud", "alpha", "compute", "tpus", "tpu-vm"]
    assert "CHECKPOINT_BUCKET" in gcp_train._remote_training_command(args)


def test_remote_command_includes_bucket():
    args = SimpleNamespace(command="run.sh", bucket="gs://foo")
    command = gcp_train._remote_training_command(args)
    assert command.startswith("export CHECKPOINT_BUCKET=gs://foo &&")
    assert command.endswith("run.sh")
