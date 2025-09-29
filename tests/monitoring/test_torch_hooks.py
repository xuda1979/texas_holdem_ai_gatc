import types

import pytest
import torch

from poker_ai.monitoring import torch_hooks


def test_attach_gradient_hooks_runs_backward_without_error():
    if not torch_hooks._HAS_TORCH:  # pragma: no cover - safety for torch-less envs
        pytest.skip("torch not available")

    model = torch.nn.Linear(2, 1)
    torch_hooks.attach_gradient_health_hooks(model)

    x = torch.tensor([[1.0, 2.0]], requires_grad=False)
    loss = model(x).sum()
    loss.backward()


def test_attach_gradient_hooks_detects_nan(monkeypatch):
    if not torch_hooks._HAS_TORCH:  # pragma: no cover - safety for torch-less envs
        pytest.skip("torch not available")

    callbacks = []

    class FakeParam:
        requires_grad = True

        def register_hook(self, fn):  # noqa: ANN001
            callbacks.append(fn)
            return types.SimpleNamespace(remove=lambda: None)

    class FakeModel:
        def named_parameters(self):  # noqa: ANN001
            return [("weight", FakeParam())]

    logged = []

    class FakeLogger:
        def log(self, name, value, extra=None):  # noqa: ANN001
            logged.append((name, value, extra))

    torch_hooks.attach_gradient_health_hooks(FakeModel(), logger=FakeLogger())

    assert callbacks, "expected register_hook to be invoked"
    with pytest.raises(FloatingPointError):
        callbacks[0](torch.tensor(float("nan")))

    assert logged == [("grad_nan_or_inf", 1.0, {"param": "weight"})]


def test_attach_gradient_hooks_noop_without_torch(monkeypatch):
    monkeypatch.setattr(torch_hooks, "_HAS_TORCH", False)

    class SentinelModel:
        def named_parameters(self):  # noqa: ANN001
            raise AssertionError("should not be called when torch is missing")

    torch_hooks.attach_gradient_health_hooks(SentinelModel())
