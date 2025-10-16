from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from poker_ai.cli import train


class DummyTrainer:
    def __init__(self) -> None:
        self.config = {"training": {}}
        self.saved_models: list[str] = []

    def save_model(self, path: str) -> None:
        self.saved_models.append(path)

    def load_model(self, path: str | None = None) -> None:  # pragma: no cover - stub
        return None


class DummySelfPlay:
    def __init__(self, *_, **__) -> None:
        self.played_iterations: list[int] = []

    def play_hand_for_training(self, iteration: int) -> None:
        self.played_iterations.append(iteration)


class DummyAnalyzer:
    def __init__(self, *_, **__) -> None:
        pass

    def on_iteration_end(self, *_, **__) -> bool:  # pragma: no cover - stub
        return True


@pytest.fixture(autouse=True)
def _stubbed_components(monkeypatch):
    monkeypatch.setattr(train, "SelfPlay", DummySelfPlay)
    monkeypatch.setattr(train, "ModelPerformanceAnalyzer", DummyAnalyzer)
    monkeypatch.setattr(train, "_find_latest_model_path", lambda *_, **__: None)


def _base_config() -> dict:
    return {
        "training": {
            "num_training_hands": 1,
            "save_model_every_samples": 0,
            "save_model_every_n_hands": 0,
            "save_model_every_minutes": 0,
            "min_buffer_before_train": 128,
        },
        "game_engine": {
            "min_players": 2,
            "max_players": 2,
            "starting_stack": 1000,
            "big_blind": 10,
            "small_blind": 5,
        },
        "curriculum": {"stages": []},
    }


def _make_args(**overrides):
    defaults = dict(
        algorithm="deep_cfr",
        gpus=False,
        npus=False,
        tpu=False,
        num_hands=None,
        save_model_every=None,
        save_minutes=None,
        save_samples=None,
        min_buffer_before_train=None,
        config=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_main(
    monkeypatch,
    args,
    torch_cuda=None,
    torch_npu=None,
    zeros_override=None,
):
    captured: dict[str, object] = {}

    def fake_parse_args():
        return args

    def fake_initialize(algorithm, config, device, *, use_data_parallel):
        captured.update(
            algorithm=algorithm,
            config=config,
            device=device,
            use_data_parallel=use_data_parallel,
        )
        return DummyTrainer()

    monkeypatch.setattr(train, "parse_args", fake_parse_args)
    monkeypatch.setattr(train, "load_configuration", lambda *_: _base_config())
    monkeypatch.setattr(train, "initialize_trainer", fake_initialize)

    if torch_cuda is not None:
        monkeypatch.setattr(train.torch, "cuda", torch_cuda, raising=False)
    if torch_npu is not None:
        monkeypatch.setattr(train.torch, "npu", torch_npu, raising=False)
    if zeros_override is not None:
        monkeypatch.setattr(train.torch, "zeros", zeros_override)

    monkeypatch.setattr(train.os, "makedirs", lambda *_, **__: None)

    train.main()
    return captured


class _CudaStub:
    def __init__(self, available: bool, count: int = 1) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _NPUStub:
    def __init__(self, available: bool, count: int = 1) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _RaisingNPUStub:
    def device_count(self) -> int:
        raise RuntimeError("device query failed")


def _install_fake_xla(monkeypatch, device: str = "xla:1") -> None:
    module = SimpleNamespace(xla_device=lambda: device)
    monkeypatch.setitem(
        sys.modules,
        "torch_xla",
        SimpleNamespace(core=SimpleNamespace(xla_model=module)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_xla.core",
        SimpleNamespace(xla_model=module),
    )
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", module)


def test_gpu_multi_device(monkeypatch, capsys):
    args = _make_args(gpus=True)
    captured = _run_main(monkeypatch, args, torch_cuda=_CudaStub(True, count=2))

    out = capsys.readouterr().out
    assert "Multi-GPU training enabled. Found 2 GPUs." in out
    assert captured["device"] == "cuda"
    assert captured["use_data_parallel"] is True


def test_gpu_fallback_to_cpu(monkeypatch, capsys):
    args = _make_args(gpus=True)
    captured = _run_main(monkeypatch, args, torch_cuda=_CudaStub(False))

    out = capsys.readouterr().out
    assert "Warning: --gpus specified, but no GPU devices are available." in out
    assert captured["device"] == "cpu"
    assert captured["use_data_parallel"] is False


def test_npu_multi_device(monkeypatch, capsys):
    args = _make_args(npus=True)
    original_zeros = train.torch.zeros

    def zeros_ignore_npu(*shape, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs = dict(kwargs)
            kwargs.pop("device", None)
        return original_zeros(*shape, **kwargs)

    captured = _run_main(
        monkeypatch,
        args,
        torch_npu=_NPUStub(True, count=3),
        zeros_override=zeros_ignore_npu,
    )

    out = capsys.readouterr().out
    assert "Multi-NPU training enabled. Found 3 NPUs." in out
    assert captured["device"] == "npu"
    assert captured["use_data_parallel"] is True


def test_npu_fallback_to_cpu(monkeypatch, capsys):
    args = _make_args(npus=True)
    captured = _run_main(monkeypatch, args, torch_npu=_NPUStub(False))

    out = capsys.readouterr().out
    assert "Warning: --npus specified, but no NPU devices are available." in out
    assert captured["device"] == "cpu"
    assert captured["use_data_parallel"] is False


def test_npu_probe_failure(monkeypatch, capsys):
    args = _make_args(npus=True)
    npu_stub = _NPUStub(True, count=2)

    original_zeros = train.torch.zeros

    def failing_zeros(*shape, **kwargs):
        if kwargs.get("device") == "npu":
            raise RuntimeError("NPU backend unavailable")
        return original_zeros(*shape, **kwargs)

    captured = _run_main(
        monkeypatch,
        args,
        torch_npu=npu_stub,
        zeros_override=failing_zeros,
    )

    out = capsys.readouterr().out
    assert "tensor allocation on the NPU backend failed" in out
    assert captured["device"] == "cpu"
    assert captured["use_data_parallel"] is False


def test_data_parallel_kwargs_for_npu(monkeypatch):
    monkeypatch.setattr(train.torch, "npu", _NPUStub(True, count=4), raising=False)

    kwargs = train._data_parallel_kwargs_for_device("npu")

    assert kwargs == {"device_ids": [0, 1, 2, 3], "output_device": 0}


def test_data_parallel_kwargs_for_npu_handles_errors(monkeypatch):
    monkeypatch.setattr(train.torch, "npu", _RaisingNPUStub(), raising=False)

    kwargs = train._data_parallel_kwargs_for_device("npu")

    assert kwargs == {}


def test_auto_selects_tpu_when_available(monkeypatch, capsys):
    _install_fake_xla(monkeypatch, device="xla:2")
    args = _make_args()
    captured = _run_main(
        monkeypatch,
        args,
        torch_cuda=_CudaStub(False),
        torch_npu=_NPUStub(False),
    )

    out = capsys.readouterr().out
    assert "Auto-selected TPU acceleration on device xla:2." in out
    assert captured["device"] == "xla"
    assert captured["use_data_parallel"] is False


def test_gpu_flag_falls_back_to_tpu(monkeypatch, capsys):
    _install_fake_xla(monkeypatch, device="xla:7")
    args = _make_args(gpus=True)
    captured = _run_main(
        monkeypatch,
        args,
        torch_cuda=_CudaStub(False),
        torch_npu=_NPUStub(False),
    )

    out = capsys.readouterr().out
    assert "Falling back to TPU acceleration on device xla:7." in out
    assert captured["device"] == "xla"
    assert captured["use_data_parallel"] is False


def test_tpu_requires_torch_xla(monkeypatch):
    args = _make_args(tpu=True)

    def fake_import(name, *args, **kwargs):
        if name == "torch_xla.core.xla_model":
            raise ImportError("no module named torch_xla")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(train, "load_configuration", lambda *_: _base_config())
    monkeypatch.setattr(train, "initialize_trainer", lambda *_, **__: DummyTrainer())
    monkeypatch.setattr(train, "SelfPlay", DummySelfPlay)
    monkeypatch.setattr(train, "ModelPerformanceAnalyzer", DummyAnalyzer)
    monkeypatch.setattr(train, "_find_latest_model_path", lambda *_, **__: None)
    monkeypatch.setattr(train.os, "makedirs", lambda *_, **__: None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        train.main()

    assert "torch_xla is required for TPU training" in str(excinfo.value)
