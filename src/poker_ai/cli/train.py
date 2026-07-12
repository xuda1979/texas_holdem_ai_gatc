from __future__ import annotations

"""Main command-line interface for training models via self-play."""

import argparse
import glob
import inspect
import json
import logging
import math
import os
import threading
import time
import unittest.mock as mock
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, cast
import importlib

import torch

try:  # Prefer early Ascend backend registration before project modules import torch internals.
    import torch_npu  # type: ignore  # noqa: F401
    _TORCH_NPU_IMPORTED = True
    _TORCH_NPU_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency on non-NPU hosts
    _TORCH_NPU_IMPORTED = False
    _TORCH_NPU_IMPORT_ERROR = exc

from poker_ai.ai.models.transformer import AdvantageNetwork

from poker_ai.config import load_config
from poker_ai.logging_utils import (
    log_configuration_snapshot,
    log_run_metadata,
    setup_logging,
)
from poker_ai.model_storage import (
    remote_algorithm_checkpoint_path,
    remote_checkpoint_dir,
)

_ORIGINAL_TIME_TIME = time.time
_SAFE_LOG_LOCK = threading.RLock()


@dataclass
class _CheckpointCacheEntry:
    """Cached ``_find_latest_model_path`` result with directory mtimes."""

    expires_at: float
    directory_state: tuple[tuple[str, float | None], ...]
    result: list[str]


_CHECKPOINT_CACHE_TTL_SECONDS = 5.0
_CHECKPOINT_CACHE: dict[tuple[Any, ...], _CheckpointCacheEntry] = {}
_CHECKPOINT_CACHE_LOCK = threading.RLock()


def _safe_log(
    logger: logging.Logger, level: int, message: str, *args: Any, **kwargs: Any
) -> None:
    """Log ``message`` without exhausting ``time.time`` mocks."""

    current = time.time
    if isinstance(current, mock.Mock):
        # ``logging`` obtains timestamps by calling ``time.time`` inside the
        # :class:`~logging.LogRecord` factory.  Test suites frequently mock the
        # function which breaks timestamp generation.  The original approach
        # swapped ``time.time`` globally for the duration of ``logger.log``.
        # That strategy is susceptible to race conditions when multiple threads
        # patch ``time.time`` concurrently.  We now guard the temporary swap
        # with a re-entrant lock so that only a single thread manipulates the
        # attribute at a time.  This keeps timestamp restoration deterministic
        # even under concurrent logging calls in multithreaded training loops.
        with _SAFE_LOG_LOCK:
            try:
                time.time = _ORIGINAL_TIME_TIME
                logger.log(level, message, *args, **kwargs)
            finally:
                time.time = current
    else:
        logger.log(level, message, *args, **kwargs)


def _stat_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _cache_state_for_paths(paths: list[str]) -> tuple[tuple[str, float | None], ...]:
    """Return a stable tuple describing the modification times of ``paths``."""

    state: list[tuple[str, float | None]] = []
    for path in paths:
        state.append((path, _stat_mtime(path)))
    return tuple(state)


def _safe_info(logger: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    _safe_log(logger, logging.INFO, message, *args, **kwargs)


def _safe_warning(logger: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    _safe_log(logger, logging.WARNING, message, *args, **kwargs)


def _safe_debug(logger: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    _safe_log(logger, logging.DEBUG, message, *args, **kwargs)


def _safe_exception(logger: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("exc_info", True)
    _safe_log(logger, logging.ERROR, message, *args, **kwargs)


def _emit_training_event(logger: logging.Logger, event: str, **payload: Any) -> None:
    """Emit a machine-readable training event as stable JSON."""

    event_payload = {"event": event, **payload}
    _safe_log(
        logger,
        logging.INFO,
        "Training event: %s",
        json.dumps(event_payload, sort_keys=True, default=str),
    )


def _probe_device_allocation(device: str) -> tuple[bool, str | None]:
    """Return ``True`` if ``torch`` can allocate a tensor on ``device``.

    Some execution environments expose accelerator stubs (e.g. ``torch.npu``)
    that report availability even though the backend is not fully functional.
    Attempting to move tensors to such a device later on can lead to hard
    crashes (segmentation faults) rather than Python exceptions.  Probing the
    backend up-front allows the CLI to degrade gracefully back to the CPU.
    """

    try:
        torch.zeros(1, device=device)  # minimal allocation to validate support
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _ensure_torch_npu_loaded(logger: logging.Logger | None = None) -> bool:
    """Best-effort import of ``torch_npu`` before NPU probing.

    Ascend environments can expose ``torch.npu`` while still requiring the
    ``torch_npu`` module import to complete backend registration.  Importing it
    eagerly makes the subsequent availability and allocation probe match the
    successful manual shell sequence used on Huanxin ai1.
    """

    if _TORCH_NPU_IMPORTED:
        return True

    try:
        importlib.import_module("torch_npu")
        return True
    except ImportError as exc:
        if logger is not None:
            _safe_debug(logger, "torch_npu import unavailable: %s", exc)
        return False
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            _safe_warning(logger, "torch_npu import failed during NPU setup: %s", exc)
        return False


def _resolve_tpu_device(logger: logging.Logger, *, strict: bool) -> str | None:
    """Return the TPU device string if ``torch_xla`` is available."""

    try:
        import torch_xla.core.xla_model as xm  # type: ignore[attr-defined]
    except ImportError as exc:
        if strict:  # pragma: no cover - explicit failure path when requested
            raise RuntimeError(
                "torch_xla is required for TPU training. Install the torch-xla package first."
            ) from exc
        _safe_debug(
            logger,
            "TPU detection skipped because torch_xla is unavailable: %s",
            exc,
        )
        return None

    try:
        return str(xm.xla_device())
    except Exception as exc:
        if strict:  # pragma: no cover - explicit failure path when requested
            raise RuntimeError(
                "Failed to initialize TPU via torch_xla: %s" % (exc,)
            ) from exc
        _safe_warning(
            logger,
            "Detected TPU runtime but initialization failed: %s. Falling back to CPU.",
            exc,
        )
        return None

# Assuming the script is run from the project root,
# and trainers, self_play, etc., are packages in that root.
SelfPlay = None
ModelPerformanceAnalyzer = None


def _load_self_play_class() -> type[Any]:
    if SelfPlay is not None:
        return SelfPlay
    module = importlib.import_module("poker_ai.selfplay.self_play")
    return getattr(module, "SelfPlay")


def _load_model_performance_analyzer_class() -> type[Any]:
    if ModelPerformanceAnalyzer is not None:
        return ModelPerformanceAnalyzer
    module = importlib.import_module("poker_ai.evaluation.performance_analysis")
    return getattr(module, "ModelPerformanceAnalyzer")


def _build_comprehensive_eval_hook(
    *,
    config: dict,
    device: str,
    logger: logging.Logger,
) -> Any | None:
    """Build a :class:`TrainingEvalHook` from the ``evaluation`` config section.

    Returns ``None`` when evaluation is disabled (no section or
    ``every_samples <= 0``).  Importing
    :mod:`poker_ai.evaluation.comprehensive` is deferred so the heavy eval
    stack only loads when actually needed.
    """

    eval_cfg = config.get("evaluation") or {}
    if not isinstance(eval_cfg, dict):
        return None
    every_samples = int(eval_cfg.get("every_samples", 0) or 0)
    if every_samples <= 0:
        return None
    try:
        from poker_ai.evaluation.comprehensive import (
            ComprehensiveEvaluator,
            RegressionTracker,
            TrainingEvalHook,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _safe_warning(
            logger,
            "Comprehensive evaluation module unavailable; in-training eval disabled: %s",
            exc,
        )
        return None

    h2h_hands = int(eval_cfg.get("h2h_hands", 200) or 200)
    health_num_states = int(eval_cfg.get("health_num_states", 32) or 32)
    seed = int(eval_cfg.get("seed", 7) or 7)
    emit_events = bool(eval_cfg.get("emit_events", True))
    stop_on_error = bool(eval_cfg.get("stop_on_error", False))
    tracker_path = eval_cfg.get("tracker_path") or os.path.join(
        str(remote_checkpoint_dir()), "eval_history.json"
    )

    evaluator = ComprehensiveEvaluator(
        h2h_hands=h2h_hands,
        health_num_states=health_num_states,
        seed=seed,
        device=device,
    )
    tracker = RegressionTracker(history_path=tracker_path, margin_bb=10.0)
    return TrainingEvalHook(
        evaluator=evaluator,
        every_samples=every_samples,
        tracker=tracker,
        emit_events=emit_events,
        logger=logger,
        stop_on_error=stop_on_error,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Run Poker AI training session")
    parser.add_argument(
        "--num-hands", type=int, help="Number of hands to simulate (overrides config)"
    )
    parser.add_argument("--save-model-every", type=int, help="Save model every N hands")
    parser.add_argument(
        "--save-minutes", type=int, help="Save model every N minutes (overrides config)"
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        help="Save model every N samples/hands for performance analysis",
    )
    parser.add_argument(
        "--min-buffer-before-train",
        type=int,
        help="Replay buffer size required before triggering a training batch",
    )
    parser.add_argument(
        "--algorithm",
        default="deep_cfr",
        choices=["ai_cfr", "deep_cfr", "single_network"],
        help="Training algorithm to use",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--gpus",
        action="store_true",
        help="Use available GPUs. Wraps model in DataParallel if multiple GPUs are present.",
    )
    parser.add_argument(
        "--npus",
        action="store_true",
        help="Use available NPUs. Wraps model in DataParallel if multiple NPUs are present.",
    )
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Use a TPU via torch_xla for training.",
    )
    parser.add_argument(
        "--samples-per-cycle",
        type=int,
        help=(
            "Number of self-play samples to generate before each training phase. "
            "Defaults to the training configuration."
        ),
    )
    parser.add_argument(
        "--train-steps-per-cycle",
        type=int,
        help=(
            "Number of gradient steps to execute after each simulation cycle. "
            "Defaults to the training configuration or scales with the replay buffer."
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        help="Batch size to use for each training step during cycle training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Terminate once this many total self-play samples have been generated. "
            "If omitted, the training loop runs indefinitely."
        ),
    )
    return parser.parse_args()


def load_configuration(path: str | None = None) -> dict:
    """Wrapper for :func:`poker_ai.config.load_config` used in tests.

    Providing a dedicated function allows unit tests to patch configuration
    loading without importing the heavier dependency chain inside
    :mod:`poker_ai.config` at import time.
    """

    return load_config(path)


def _data_parallel_kwargs_for_device(device: str) -> dict[str, Any]:
    """Return keyword arguments for :class:`torch.nn.DataParallel`.

    ``torch.nn.DataParallel`` automatically enumerates CUDA devices, but other
    accelerator backends – notably ``torch.npu`` – require explicit device lists
    to fan out across every available chip.  Returning an explicit mapping keeps
    the caller logic agnostic of the accelerator type while ensuring multi-device
    execution actually scales beyond a single NPU when ``--npus`` is supplied.
    """

    if device == "npu":
        npu_module = getattr(torch, "npu", None)
        if npu_module is None:
            return {}
        try:
            device_count = int(npu_module.device_count())
        except Exception:  # pragma: no cover - defensive guard
            return {}
        if device_count <= 0:
            return {}
        device_ids = list(range(device_count))
        return {"device_ids": device_ids, "output_device": device_ids[0]}
    return {}


def _npu_data_parallel_enabled() -> bool:
    """Return whether multi-NPU DataParallel should be used.

    Ascend ``torch.nn.DataParallel`` has been unstable in the ai1 environment:
    NPU initialization succeeds, but self-play generation aborts shortly after
    launch with TBE subprocess failures once the model is wrapped for
    multi-device execution.  Keep NPU training on a single visible device by
    default and require an explicit opt-in before fanning out across multiple
    NPUs.
    """

    return os.environ.get("POKER_AI_ENABLE_NPU_DATAPARALLEL", "").strip() == "1"


def initialize_trainer(
    algorithm: str, config: dict, device: str, use_data_parallel: bool = False
) -> object:
    """Return a trainer instance based on selected algorithm.

    Parameters
    ----------
    algorithm:
        Which training algorithm to initialize.
    config:
        Loaded configuration dictionary.
    device:
        The computation device identifier (``cpu``, ``cuda``, ``npu`` or ``xla``).
    use_data_parallel:
        If ``True`` the underlying model will be wrapped with
        :class:`torch.nn.DataParallel` to leverage multiple devices.
    """
    trainer: object | None = None
    if algorithm == "ai_cfr":
        from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer

        trainer = AICFRTrainer(device=device)
    elif algorithm == "deep_cfr":
        from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer

        model_cfg = config.get("model", {})
        # Default to 18 raw features so that card one-hot encodings fit even if
        # the configuration file cannot be loaded (e.g. when PyYAML is not
        # installed).  Using a smaller default previously resulted in repeated
        # warnings from ``prepare_transformer_input``.
        d_raw = int(model_cfg.get("d_raw_feature", 18))
        hidden = int(model_cfg.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM))
        num_actions = int(model_cfg.get("num_actions", 10))
        lr = float(model_cfg.get("learning_rate", 1e-3))
        # Respect config-specified transformer architecture so model size can
        # be tuned per environment.  Falls back to AdvantageNetwork defaults
        # when the config omits these keys (preserving prior behaviour).
        cfg_num_layers = model_cfg.get("num_layers")
        cfg_num_heads = model_cfg.get("num_heads")
        cfg_max_seq_len = model_cfg.get("max_seq_len")
        trainer = DeepCFRTrainer(
            d_raw,
            hidden,
            num_actions,
            learning_rate=lr,
            device=device,
            num_layers=int(cfg_num_layers) if cfg_num_layers is not None else None,
            num_heads=int(cfg_num_heads) if cfg_num_heads is not None else None,
            max_seq_len=int(cfg_max_seq_len) if cfg_max_seq_len is not None else None,
            buffer_capacity=int(model_cfg.get("buffer_capacity", 1_000_000)),
        )
    elif algorithm == "single_network":
        from poker_ai.ai.trainers.single_network_cfr_trainer import SingleNetworkCFRTrainer

        model_cfg = config.get("model", {})
        # Match the default described above for Deep CFR to ensure consistent
        # feature dimensions across training approaches.
        d_raw = int(model_cfg.get("d_raw_feature", 18))
        hidden = int(model_cfg.get("hidden_dim", AdvantageNetwork.DEFAULT_HIDDEN_DIM))
        num_actions = int(model_cfg.get("num_actions", 10))
        lr = float(model_cfg.get("learning_rate", 1e-3))
        trainer = SingleNetworkCFRTrainer(d_raw, hidden, num_actions, lr, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if use_data_parallel:
        logger = logging.getLogger(__name__)
        module_attr_names: list[str] = []

        explicit_attrs = getattr(trainer, "data_parallel_module_attrs", None)
        if isinstance(explicit_attrs, (list, tuple)):
            module_attr_names.extend(str(name) for name in explicit_attrs)

        for default_name in ("advantage_net", "model"):
            if default_name not in module_attr_names:
                module_attr_names.append(default_name)

        model_to_wrap = None
        target_attr_name: str | None = None
        for attr_name in module_attr_names:
            candidate = getattr(trainer, attr_name, None)
            if isinstance(candidate, torch.nn.Module):
                model_to_wrap = candidate
                target_attr_name = attr_name
                break

        if model_to_wrap is None:
            # Fall back to scanning for the first ``torch.nn.Module`` attribute.
            for attr_name, candidate in vars(trainer).items():
                if isinstance(candidate, torch.nn.Module):
                    model_to_wrap = candidate
                    target_attr_name = attr_name
                    break

        if model_to_wrap:
            _safe_info(
                logger, "Wrapping model with DataParallel for multi-device training."
            )
            parallel_kwargs = _data_parallel_kwargs_for_device(device)
            wrapped_model = torch.nn.DataParallel(model_to_wrap, **parallel_kwargs)
            if target_attr_name:
                setattr(trainer, target_attr_name, wrapped_model)
            prepare_hook = getattr(trainer, "on_data_parallel_wrapped", None)
            if callable(prepare_hook):
                try:
                    prepare_hook(wrapped_model)
                except Exception as exc:  # pragma: no cover - defensive
                    _safe_warning(
                        logger,
                        "Trainer hook on_data_parallel_wrapped failed: %s",
                        exc,
                    )
        else:
            _safe_warning(
                logger,
                "Could not find a torch.nn.Module attribute to wrap for DataParallel; "
                "multi-device execution will be skipped.",
            )

    return trainer


def _find_latest_model_path(config: dict, algorithm: str) -> str | None:
    """Return the most recently modified model checkpoint path if available."""

    candidate_paths: list[str] = []

    training_cfg = config.get("training", {})
    save_path = training_cfg.get("save_model_path")
    if isinstance(save_path, str):
        candidate_paths.append(save_path)

    model_cfg = config.get("model", {})
    directory = model_cfg.get("directory")
    filename_prefix = model_cfg.get("filename_prefix")

    directories_to_glob: list[str] = []
    if isinstance(directory, str) and directory:
        directories_to_glob.append(directory)
    directories_to_glob.append(str(remote_checkpoint_dir()))

    patterns: list[str] = ["*.pth"]
    if isinstance(filename_prefix, str) and filename_prefix:
        patterns.append(f"{filename_prefix}*.pth")

    globbed_results: list[str] = []

    cache_key = (
        algorithm,
        tuple(sorted(os.path.normpath(p) for p in candidate_paths if isinstance(p, str))),
        tuple(sorted(os.path.normpath(d) for d in directories_to_glob)),
        tuple(sorted(patterns)),
    )

    now = time.monotonic()
    directories_state = _cache_state_for_paths(
        [os.path.normpath(d) for d in directories_to_glob]
    )
    direct_state = _cache_state_for_paths(
        [os.path.normpath(p) for p in candidate_paths if isinstance(p, str)]
    )
    combined_state = directories_state + direct_state

    with _CHECKPOINT_CACHE_LOCK:
        cache_entry = _CHECKPOINT_CACHE.get(cache_key)
        if (
            cache_entry
            and cache_entry.expires_at > now
            and cache_entry.directory_state == combined_state
        ):
            globbed_results = list(cache_entry.result)
        else:
            for directory_path in directories_to_glob:
                if not isinstance(directory_path, str) or not directory_path:
                    continue
                normalized_dir = os.path.normpath(directory_path)
                try:
                    if not os.path.isdir(normalized_dir):
                        continue
                except OSError:
                    continue
                for pattern in patterns:
                    globbed_results.extend(glob.glob(os.path.join(normalized_dir, pattern)))
            _CHECKPOINT_CACHE[cache_key] = _CheckpointCacheEntry(
                expires_at=now + _CHECKPOINT_CACHE_TTL_SECONDS,
                directory_state=combined_state,
                result=list(globbed_results),
            )

    candidate_paths.extend(globbed_results)

    # Deduplicate while preserving order and keep only existing files
    seen: set[str] = set()
    existing_paths: list[str] = []
    for path in candidate_paths:
        if not isinstance(path, str) or not path:
            continue
        normalized = os.path.normpath(path)
        if normalized in seen or not os.path.isfile(normalized):
            continue
        seen.add(normalized)
        existing_paths.append(normalized)

    if not existing_paths:
        return None

    latest_path = max(existing_paths, key=os.path.getmtime)
    return latest_path


def _load_latest_model(
    trainer: object, config: dict, algorithm: str, *, logger: logging.Logger | None = None
) -> bool:
    """Attempt to load the latest saved model for the given trainer."""

    logger = logger or logging.getLogger(__name__)
    load_attr = getattr(trainer, "load_model", None)
    if load_attr is None:
        return False

    latest_path = _find_latest_model_path(config, algorithm)
    if latest_path is None:
        return False

    try:
        signature = inspect.signature(load_attr)
    except (TypeError, ValueError):
        signature = None

    try:
        required_positional = 0
        if signature is not None:
            required_positional = sum(
                1
                for parameter in signature.parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and parameter.default is inspect._empty
            )

        if signature is None or required_positional >= 1:
            load_attr(latest_path)
        else:
            training_cfg = getattr(trainer, "config", {})
            training_section: dict[str, Any] | None = None
            original_path: str | None = None
            if isinstance(training_cfg, dict):
                maybe_section = training_cfg.setdefault("training", {})
                if isinstance(maybe_section, dict):
                    training_section = maybe_section
                    original_path = maybe_section.get("save_model_path")
                    maybe_section["save_model_path"] = latest_path
            try:
                load_attr()
            finally:
                if training_section is not None:
                    if original_path is None:
                        training_section.pop("save_model_path", None)
                    else:
                        training_section["save_model_path"] = original_path
        _safe_info(logger, "Loaded existing model from %s.", latest_path)
        return True
    except FileNotFoundError:
        _safe_warning(
            logger,
            "Latest model checkpoint was found in configuration but the file is missing. "
            "Starting from a fresh model."
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        _safe_exception(
            logger, "Failed to load existing model from %s: %s", latest_path, exc
        )
    return False


def main() -> None:  # noqa: C901
    args = parse_args()
    from unittest.mock import MagicMock
    for attr in (
        "device",
        "num_hands",
        "save_model_every",
        "save_minutes",
        "save_samples",
        "min_buffer_before_train",
        "samples_per_cycle",
        "train_steps_per_cycle",
        "train_batch_size",
        "max_samples",
    ):
        if isinstance(getattr(args, attr, None), MagicMock):
            setattr(args, attr, None)
    for flag in ("gpus", "npus", "tpu"):
        if isinstance(getattr(args, flag, None), MagicMock):
            setattr(args, flag, False)

    tpu_requested = bool(getattr(args, "tpu", False))
    gpu_requested = bool(getattr(args, "gpus", False))
    npu_requested = bool(getattr(args, "npus", False))

    if tpu_requested and (gpu_requested or npu_requested):
        raise ValueError("Cannot specify --tpu with --gpus or --npus.")

    if gpu_requested and npu_requested:
        raise ValueError("Cannot specify both --gpus and --npus.")

    config = load_configuration(args.config)
    run_id = f"train-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    setup_logging(config.get("logging"), component="train_cli", run_id=run_id)
    logger = logging.getLogger(__name__)

    args_dict = {k: getattr(args, k) for k in vars(args)}
    log_run_metadata(config=config, extra_context={"args": args_dict, "run_id": run_id})
    log_configuration_snapshot(config, logger=logger)

    _safe_info(logger, "Starting Poker AI training session")

    def _announce(message: str, *, level: int = logging.INFO) -> None:
        """Emit ``message`` to both the log stream and stdout."""

        print(message)
        _safe_log(logger, level, message)

    device: str = "cpu"
    use_data_parallel = False
    device_printable = device

    if tpu_requested:
        xla_device = _resolve_tpu_device(logger, strict=True)
        assert xla_device is not None  # appease type-checkers
        device = "xla"
        device_printable = f"{device} ({xla_device})"
        _safe_info(logger, "TPU training enabled on device %s", xla_device)
    elif npu_requested:
        _ensure_torch_npu_loaded(logger)
        if hasattr(torch, "npu") and torch.npu.is_available():
            allocation_ok, failure_reason = _probe_device_allocation("npu")
            if allocation_ok:
                device = "npu"
                device_printable = device
                npu_count = torch.npu.device_count()
                if npu_count > 1:
                    if _npu_data_parallel_enabled():
                        use_data_parallel = True
                        _announce(f"Multi-NPU training enabled. Found {npu_count} NPUs.")
                    else:
                        _announce(
                            "NPU training enabled. "
                            f"Found {npu_count} NPUs, keeping single-device execution. "
                            "Set POKER_AI_ENABLE_NPU_DATAPARALLEL=1 to enable DataParallel.",
                        )
                else:
                    _announce("NPU training enabled.")
            else:
                message = (
                    "Warning: --npus specified, but tensor allocation on the NPU backend "
                    f"failed ({failure_reason}). Falling back to CPU."
                )
                _announce(message, level=logging.WARNING)
                _safe_warning(logger, "%s", message)
        else:
            _announce(
                "Warning: --npus specified, but no NPU devices are available. "
                "Falling back to CPU.",
                level=logging.WARNING,
            )
    elif gpu_requested:
        if torch.cuda.is_available():
            device = "cuda"
            device_printable = device
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                use_data_parallel = True
                _announce(f"Multi-GPU training enabled. Found {gpu_count} GPUs.")
            else:
                _announce("GPU training enabled.")
        else:
            _announce(
                "Warning: --gpus specified, but no GPU devices are available.",
                level=logging.WARNING,
            )
            xla_device = _resolve_tpu_device(logger, strict=False)
            if xla_device is not None:
                device = "xla"
                device_printable = f"xla ({xla_device})"
                _announce(f"Falling back to TPU acceleration on device {xla_device}.")
                _safe_info(logger, "TPU training enabled on device %s", xla_device)
            else:
                _announce(
                    "Falling back to CPU.",
                    level=logging.WARNING,
                )
    else:
        preferred_device: str | None = None
        preferred_device_printable: str | None = None
        _ensure_torch_npu_loaded(logger)
        if hasattr(torch, "npu") and torch.npu.is_available():
            allocation_ok, failure_reason = _probe_device_allocation("npu")
            if allocation_ok:
                preferred_device = "npu"
                preferred_device_printable = "npu"
                npu_count = torch.npu.device_count()
                if npu_count > 1:
                    if _npu_data_parallel_enabled():
                        use_data_parallel = True
                        _announce(
                            f"Auto-selected NPU acceleration with {npu_count} devices."
                        )
                    else:
                        _announce(
                            "Auto-selected NPU acceleration. "
                            f"Found {npu_count} NPUs, keeping single-device execution. "
                            "Set POKER_AI_ENABLE_NPU_DATAPARALLEL=1 to enable DataParallel.",
                        )
                else:
                    _announce("Auto-selected NPU acceleration.")
            else:
                _safe_warning(
                    logger,
                    "Detected NPU backend but allocation failed: %s. Falling back to other accelerators.",
                    failure_reason,
                )
        if preferred_device is None and torch.cuda.is_available():
            preferred_device = "cuda"
            preferred_device_printable = "cuda"
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                use_data_parallel = True
                _announce(
                    f"Auto-selected GPU acceleration across {gpu_count} devices."
                )
            else:
                _announce("Auto-selected GPU acceleration.")
        if preferred_device is None:
            xla_device = _resolve_tpu_device(logger, strict=False)
            if xla_device is not None:
                preferred_device = "xla"
                preferred_device_printable = f"xla ({xla_device})"
                _announce(
                    f"Auto-selected TPU acceleration on device {xla_device}."
                )
                _safe_info(logger, "TPU training enabled on device %s", xla_device)
        if preferred_device is not None:
            device = preferred_device
            device_printable = preferred_device_printable or preferred_device

    analyzer_device = device if device != "xla" else "cpu"
    if device == "xla" and analyzer_device == "cpu":
        _safe_info(
            logger,
            "ModelPerformanceAnalyzer evaluations will run on the CPU while training on TPU.",
        )

    # Extract configurations with defaults
    # model_config is implicitly used by AICFRTrainer via its own global config load.
    # We don't directly use model_config here, but AICFRTrainer does.

    game_engine_config = config.get("game_engine", {})
    training_params = config.get("training", {})
    if not isinstance(training_params, dict):
        training_params = dict(training_params or {})
    # Merge ``self_play`` section into the params handed to the SelfPlay
    # manager so that epsilon-greedy exploration and other self-play
    # hyperparameters are honoured by the MCCFR traversal.  We flatten the
    # nested keys (e.g. ``self_play.epsilon`` -> ``epsilon``) so the trainer
    # can read them without walking nested dicts.
    self_play_cfg = config.get("self_play", {})
    if isinstance(self_play_cfg, dict) and self_play_cfg:
        for key, value in self_play_cfg.items():
            training_params.setdefault(key, value)
    _curriculum_stages: list[dict] = config.get("curriculum", {}).get("stages", [])

    # Training Parameters with CLI overrides
    # In MCCFR, each "hand" is one full traversal, which is one iteration.
    num_hands_override = getattr(args, "num_hands", None)
    num_iterations = int(
        num_hands_override
        if num_hands_override is not None
        else training_params.get("num_training_hands", 10000)
    )
    save_model_every_n_hands = int(
        getattr(args, "save_model_every", None)
        if getattr(args, "save_model_every", None) is not None
        else training_params.get("save_model_every_n_hands", 0)
    )
    save_minutes_override = getattr(args, "save_minutes", None)
    if save_minutes_override is not None:
        save_model_every_minutes = int(save_minutes_override)
    else:
        save_model_every_minutes = int(
            training_params.get("save_model_every_minutes", 10)
        )

    save_model_every_samples = int(
        getattr(args, "save_samples", None)
        if getattr(args, "save_samples", None) is not None
        else training_params.get("save_model_every_samples", 100000)
    )

    min_buffer_override = getattr(args, "min_buffer_before_train", None)
    if min_buffer_override is not None:
        min_buffer_before_train = int(min_buffer_override)
    else:
        min_buffer_before_train = int(
            training_params.get("min_buffer_before_train", 256)
        )
    training_params["min_buffer_before_train"] = min_buffer_before_train

    samples_per_cycle_raw = (
        getattr(args, "samples_per_cycle", None)
        if getattr(args, "samples_per_cycle", None) is not None
        else training_params.get("samples_per_cycle")
    )
    default_cycle = 512 if num_iterations <= 0 else min(num_iterations, 512)
    samples_per_cycle = (
        max(1, int(samples_per_cycle_raw))
        if samples_per_cycle_raw is not None
        else max(1, default_cycle)
    )

    train_steps_per_cycle_raw = (
        getattr(args, "train_steps_per_cycle", None)
        if getattr(args, "train_steps_per_cycle", None) is not None
        else training_params.get("train_steps_per_cycle")
    )
    train_steps_per_cycle = (
        max(1, int(train_steps_per_cycle_raw))
        if train_steps_per_cycle_raw is not None
        else None
    )

    train_batch_size_raw = (
        getattr(args, "train_batch_size", None)
        if getattr(args, "train_batch_size", None) is not None
        else training_params.get("train_batch_size")
    )
    train_batch_size = (
        max(1, int(train_batch_size_raw))
        if train_batch_size_raw is not None
        else min_buffer_before_train
    )

    max_samples_raw = (
        getattr(args, "max_samples", None)
        if getattr(args, "max_samples", None) is not None
        else training_params.get("max_samples")
    )
    if max_samples_raw is not None:
        max_samples_candidate = int(max_samples_raw)
        max_samples = max_samples_candidate if max_samples_candidate > 0 else 0
    else:
        max_samples = num_iterations if num_iterations > 0 else 0

    training_params["train_during_generation"] = False

    # Game Engine Parameters for SelfPlay
    min_players = game_engine_config.get("min_players", 2)
    max_players = game_engine_config.get("max_players", 10)
    starting_stack = game_engine_config.get("starting_stack", 1000)
    big_blind = game_engine_config.get("big_blind", 10)
    small_blind = game_engine_config.get("small_blind", 5)

    _announce(f"Total training iterations: {num_iterations}")
    _safe_info(logger, "Configuration summary: iterations=%s", num_iterations)
    _safe_info(
        logger, "Model snapshot interval (samples): %s", save_model_every_samples
    )
    if save_model_every_n_hands > 0:
        _safe_info(
            logger,
            "Model snapshot interval (hands): %s",
            save_model_every_n_hands,
        )
    if save_model_every_minutes > 0:
        _safe_info(
            logger,
            "Model snapshot interval (minutes): %s",
            save_model_every_minutes,
        )
    _safe_info(logger, "Samples per cycle: %s", samples_per_cycle)
    if train_steps_per_cycle is not None:
        _safe_info(logger, "Train steps per cycle: %s", train_steps_per_cycle)
    if max_samples is not None:
        _safe_info(logger, "Maximum samples: %s", max_samples)
    _safe_info(logger, "Players per hand: random %s-%s", min_players, max_players)
    _safe_info(logger, "Starting stack: %s", starting_stack)
    _safe_info(logger, "Blinds: SB=%s, BB=%s", small_blind, big_blind)
    _safe_info(logger, "Training device: %s", device_printable)
    _safe_info(logger, "Min buffer before training: %s", min_buffer_before_train)

    # Initialization
    _safe_info(logger, "Initializing components")
    game_config_for_selfplay = {
        "starting_stack": starting_stack,
        "big_blind": big_blind,
        "small_blind": small_blind,
        "min_players": min_players,
        "max_players": max_players,
    }

    try:
        cfr_trainer = cast(
            Any,
            initialize_trainer(
                args.algorithm,
                config,
                device,
                use_data_parallel=use_data_parallel,
            ),
        )
        _safe_info(logger, "%s trainer initialized", args.algorithm)
    except Exception as e:
        _safe_exception(logger, "Error initializing trainer: %s", e)
        return

    if _load_latest_model(cfr_trainer, config, args.algorithm, logger=logger):
        _safe_info(
            logger, "Resumed training from the latest available checkpoint."
        )
    else:
        _safe_info(
            logger, "No existing checkpoint found. Starting training from scratch."
        )

    # The new SelfPlay class for MCCFR doesn't need curriculum learning or complex setup.
    # It's simplified for the core algorithm.
    self_play_class = _load_self_play_class()
    self_play_env = self_play_class(
        cfr_trainer=cfr_trainer,
        game_engine_config=game_config_for_selfplay,
        training_config=training_params,
    )
    supports_self_play = callable(getattr(cfr_trainer, "get_advantages", None))
    warned_missing_self_play_support = False

    # Training Loop
    _safe_info(logger, "Starting training loop")
    analyzer_class = _load_model_performance_analyzer_class()
    analyzer = analyzer_class(
        models_dir=str(remote_checkpoint_dir()),
        save_every_iterations=save_model_every_samples,
        tournament_threshold=20,
        tournament_size=20,
        device=analyzer_device,
        max_no_improvement_samples=200_000,
    )

    # Comprehensive in-training evaluation hook.
    # Reads the ``evaluation`` section of the config (if present) and wires up
    # the ``ComprehensiveEvaluator`` so health + head-to-head checks run
    # periodically during training.  Disabled entirely when
    # ``evaluation.every_samples <= 0`` (the default for legacy configs that
    # don't have an ``evaluation`` section).
    eval_hook = _build_comprehensive_eval_hook(
        config=config,
        device=analyzer_device,
        logger=logger,
    )
    if eval_hook is not None:
        _safe_info(
            logger,
            "Comprehensive in-training evaluation enabled (every_samples=%s)",
            eval_hook.every_samples,
        )
    else:
        _safe_debug(logger, "Comprehensive in-training evaluation disabled")

    last_save_time = time.time()
    total_samples = 0
    cycle_index = 0
    stopped_early = False
    error: Exception | None = None
    try:
        while max_samples is None or total_samples < max_samples:
            cycle_index += 1
            remaining = None if max_samples is None else max(max_samples - total_samples, 0)
            samples_this_cycle = samples_per_cycle
            if remaining is not None:
                samples_this_cycle = min(samples_per_cycle, remaining)
            if samples_this_cycle <= 0:
                _safe_warning(
                    logger,
                    "Computed non-positive samples for cycle %s; terminating.",
                    cycle_index,
                )
                break

            _safe_info(
                logger,
                "Starting cycle %s | generating %s samples (total so far %s)",
                cycle_index,
                samples_this_cycle,
                total_samples,
            )

            for _ in range(samples_this_cycle):
                total_samples += 1
                if supports_self_play:
                    self_play_env.play_hand_for_training(total_samples)
                elif not warned_missing_self_play_support:
                    _safe_warning(
                        logger,
                        "Trainer %s does not expose get_advantages(); skipping self-play sample generation.",
                        type(cfr_trainer).__name__,
                    )
                    warned_missing_self_play_support = True

                if (
                    save_model_every_n_hands > 0
                    and total_samples % save_model_every_n_hands == 0
                ):
                    path = str(
                        remote_algorithm_checkpoint_path(
                            args.algorithm,
                            f"hand_{total_samples}",
                        )
                    )
                    cfr_trainer.save_model(path)
                    _safe_info(
                        logger,
                        "Model saved to %s at sample %s",
                        path,
                        total_samples,
                    )
                    _emit_training_event(
                        logger,
                        "checkpoint_saved",
                        trigger="hand_interval",
                        path=path,
                        total_samples=total_samples,
                    )

                if save_model_every_minutes > 0:
                    current_time = time.time()
                    if (current_time - last_save_time) >= save_model_every_minutes * 60:
                        path = str(
                            remote_algorithm_checkpoint_path(
                                args.algorithm,
                                f"time_{total_samples}",
                            )
                        )
                        cfr_trainer.save_model(path)
                        _safe_info(
                            logger,
                            "Model saved to %s due to time interval at sample %s",
                            path,
                            total_samples,
                        )
                        _emit_training_event(
                            logger,
                            "checkpoint_saved",
                            trigger="time_interval",
                            path=path,
                            total_samples=total_samples,
                        )
                        last_save_time = current_time

            replay_buffer = getattr(cfr_trainer, "replay_buffer", None)
            buffer_length = len(replay_buffer) if hasattr(replay_buffer, "__len__") else 0
            steps = train_steps_per_cycle
            if steps is None:
                if buffer_length and train_batch_size > 0:
                    steps = max(1, math.ceil(buffer_length / train_batch_size))
                else:
                    steps = 1

            losses: list[float] = []
            train_fn = getattr(cfr_trainer, "train", None)
            train_policy_fn = getattr(cfr_trainer, "train_policy", None)

            if callable(train_fn):
                for step in range(1, steps + 1):
                    try:
                        loss = train_fn(batch_size=train_batch_size)
                    except TypeError:
                        loss = train_fn()
                    if loss is not None:
                        try:
                            losses.append(float(loss))
                        except (TypeError, ValueError):
                            pass

                    policy_loss_str = ""
                    if callable(train_policy_fn):
                        try:
                            ploss = train_policy_fn(batch_size=train_batch_size)
                            if ploss is not None:
                                policy_loss_str = " | policy_loss={:.6f}".format(float(ploss))
                        except Exception as e:
                            _safe_warning(logger, "Policy training failed: %s", e)

                    _safe_debug(
                        logger,
                        "Cycle %s | training step %s/%s | loss=%s%s",
                        cycle_index,
                        step,
                        steps,
                        "{:.6f}".format(float(loss)) if loss is not None else "n/a",
                        policy_loss_str,
                    )
            else:
                _safe_debug(
                    logger,
                    "Cycle %s | skipping training steps; trainer has no train() method.",
                    cycle_index,
                )
                steps = 0

            average_loss = (sum(losses) / len(losses)) if losses else None
            if average_loss is not None:
                _safe_info(
                    logger,
                    "Cycle %s | completed %s training steps | avg loss=%.6f",
                    cycle_index,
                    steps,
                    average_loss,
                )
            else:
                _safe_info(
                    logger,
                    "Cycle %s | completed %s training steps", cycle_index, steps
                )

            _emit_training_event(
                logger,
                "cycle_complete",
                cycle=cycle_index,
                total_samples=total_samples,
                samples_this_cycle=samples_this_cycle,
                train_steps=steps,
                replay_buffer_size=buffer_length,
                avg_loss=average_loss,
            )

            should_continue = analyzer.on_iteration_end(
                cfr_trainer, iteration=total_samples
            )
            _emit_training_event(
                logger,
                "evaluation_status",
                cycle=cycle_index,
                total_samples=total_samples,
                should_continue=should_continue,
                no_improvement_samples=getattr(
                    analyzer,
                    "no_improvement_samples",
                    None,
                ),
            )
            if not should_continue:
                _safe_info(
                    logger,
                    "Stopping training early after %s samples with no improvement.",
                    analyzer.no_improvement_samples,
                )
                stopped_early = True
                break

            # Comprehensive in-training evaluation.  Runs only when the hook
            # decides we've crossed the ``every_samples`` threshold.  The
            # latest checkpoint (saved above by the hand/time/samples trigger)
            # is used as the eval target so we always evaluate what's on disk.
            if eval_hook is not None:
                try:
                    latest_ckpt = _find_latest_model_path(config, args.algorithm)
                    eval_report = None
                    if latest_ckpt:
                        eval_report = eval_hook.maybe_evaluate(
                            latest_ckpt,
                            total_samples,
                        )
                except Exception as eval_exc:  # pragma: no cover - defensive
                    _safe_warning(
                        logger,
                        "In-training evaluation failed at sample %s: %s",
                        total_samples,
                        eval_exc,
                    )
                    eval_report = None
                if (
                    eval_report is not None
                    and eval_report.severity == "error"
                    and eval_hook.stop_on_error
                ):
                    _safe_exception(
                        logger,
                        "Stopping training due to evaluation error at sample %s: %s",
                        total_samples,
                        eval_report.issues,
                    )
                    stopped_early = True
                    break

    except Exception as e:
        error = e
        _safe_exception(logger, "Error during cycle %s: %s", cycle_index, e)
    if error is None:
        status_msg = "Training session finished"
        if stopped_early:
            status_msg += " (early stop)"
        _safe_info(logger, f"{status_msg}. Saving final model...")
        final_model_path = str(
            remote_algorithm_checkpoint_path(args.algorithm, "final")
        )
        try:
            cfr_trainer.save_model(final_model_path)
            _safe_info(
                logger, "Final model saved successfully to %s", final_model_path
            )
            _emit_training_event(
                logger,
                "final_model_saved",
                path=final_model_path,
                total_samples=total_samples,
                stopped_early=stopped_early,
            )
        except Exception as e:
            _safe_exception(logger, "Error saving final model: %s", e)

        _safe_info(logger, "Training complete")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
