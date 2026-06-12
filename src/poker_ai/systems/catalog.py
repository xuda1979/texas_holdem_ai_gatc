"""Static subsystem catalog for targeted diagnostics and fast iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Iterable, Sequence


def _normalise_path(path: str) -> str:
    candidate = str(PurePath(path)).replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    return candidate


def _path_matches_prefix(path: str, prefix: str) -> bool:
    candidate = _normalise_path(path)
    target = _normalise_path(prefix).rstrip("/")
    return candidate == target or candidate.startswith(f"{target}/")


@dataclass(frozen=True)
class SubsystemSpec:
    """Describe a logical subsystem and the checks that protect it."""

    name: str
    code_paths: tuple[str, ...]
    test_targets: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    focus: str = ""
    criticality: str = "medium"

    def owns_path(self, path: str) -> bool:
        prefixes = (*self.code_paths, *self.test_targets)
        return any(_path_matches_prefix(path, prefix) for prefix in prefixes)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code_paths": list(self.code_paths),
            "test_targets": list(self.test_targets),
            "dependencies": list(self.dependencies),
            "diagnostics": list(self.diagnostics),
            "focus": self.focus,
            "criticality": self.criticality,
        }


DEFAULT_SUBSYSTEM_SPECS: tuple[SubsystemSpec, ...] = (
    SubsystemSpec(
        name="config-runtime",
        code_paths=(
            "pyproject.toml",
            "requirements.txt",
            "src/poker_ai/config",
        ),
        test_targets=(
            "tests/hygiene/test_config_schema.py",
            "tests/test_pyproject_metadata.py",
            "tests/cli/test_train_devices.py",
        ),
        diagnostics=("config-load",),
        focus="Configuration loading, dependency metadata, and runtime defaults.",
        criticality="high",
    ),
    SubsystemSpec(
        name="rules-engine",
        code_paths=(
            "src/poker_ai/engine",
            "src/poker_ai/rules",
            "src/gatc_poker",
        ),
        test_targets=(
            "tests/engine",
            "tests/rules",
            "tests/test_rules.py",
            "tests/test_handeval.py",
            "tests/test_side_pots.py",
            "tests/test_texas_holdem_rules.py",
            "tests/test_action_legality_hypothesis.py",
        ),
        diagnostics=("hand-eval", "action-legality"),
        focus="Core Hold'em state transitions, payouts, side pots, and legality.",
        criticality="high",
    ),
    SubsystemSpec(
        name="cfr-algorithms",
        code_paths=(
            "src/cfr_trainer.py",
            "src/poker_ai/ai/trainers",
            "src/poker_ai/systems/cfr.py",
            "src/poker_ai/rules/cfr.py",
        ),
        test_targets=(
            "tests/cfr",
            "tests/deepcfr",
            "tests/test_ai_cfr_trainer.py",
            "tests/test_cfr.py",
            "tests/test_cfr_plus.py",
            "tests/test_cfr_weighting.py",
            "tests/test_trainer_weighted_loss.py",
        ),
        dependencies=("rules-engine", "models-embeddings", "self-play-runtime"),
        diagnostics=("forward-pass", "training-step"),
        focus="Regret minimization trainers and advantage learning loops.",
        criticality="high",
    ),
    SubsystemSpec(
        name="models-embeddings",
        code_paths=(
            "src/poker_ai/ai/models",
            "src/poker_ai/utils/state_representation.py",
            "src/poker_ai/systems/models.py",
            "src/poker_ai/systems/embeddings.py",
            "src/poker_ai/utils/action_mapping.py",
        ),
        test_targets=(
            "tests/test_model_forward_pass.py",
            "tests/test_state_representation.py",
            "tests/test_action_mapping.py",
            "tests/test_systems_components.py",
        ),
        dependencies=("rules-engine",),
        diagnostics=("forward-pass",),
        focus="State encoding, sequence embeddings, and neural inference surfaces.",
        criticality="high",
    ),
    SubsystemSpec(
        name="self-play-runtime",
        code_paths=(
            "src/poker_ai/selfplay",
            "src/poker_ai/cli/self_play.py",
            "src/poker_ai/systems/self_play.py",
        ),
        test_targets=(
            "tests/selfplay",
            "tests/test_self_play.py",
            "tests/test_flaky_simulation.py",
        ),
        dependencies=("rules-engine", "cfr-algorithms"),
        diagnostics=("self-play",),
        focus="Simulation, replay buffer population, and worker orchestration.",
        criticality="high",
    ),
    SubsystemSpec(
        name="training-orchestration",
        code_paths=(
            "src/poker_ai/cli/train.py",
            "src/poker_ai/systems/training.py",
            "training",
            "run_training.py",
        ),
        test_targets=(
            "tests/test_cli_train.py",
            "tests/cli/test_train_devices.py",
            "tests/test_training_subsystem.py",
            "tests/test_train_cli_observability_unittest.py",
            "tests/test_run_training.py",
        ),
        dependencies=("cfr-algorithms", "self-play-runtime", "monitoring-observability"),
        diagnostics=("forward-pass", "training-step", "self-play"),
        focus="Top-level train loops, device selection, and cycle observability.",
        criticality="high",
    ),
    SubsystemSpec(
        name="evaluation-analytics",
        code_paths=(
            "src/poker_ai/evaluation",
            "src/poker_ai/systems/evaluation.py",
        ),
        test_targets=(
            "tests/eval",
            "tests/evaluation",
            "tests/test_performance_analysis.py",
            "tests/test_exploitability.py",
        ),
        dependencies=("cfr-algorithms", "rules-engine"),
        diagnostics=("exploitability", "tournament"),
        focus="Exploitability, evaluation harnesses, and benchmark reports.",
        criticality="medium",
    ),
    SubsystemSpec(
        name="model-storage",
        code_paths=(
            "src/poker_ai/model_storage.py",
            "src/poker_ai/utils/model_paths.py",
            "src/poker_ai/ai/model_loader.py",
            "src/poker_ai/evaluation/ai_gto_analyzer.py",
        ),
        test_targets=(
            "tests/test_model_loader.py",
            "tests/test_model_paths.py",
            "tests/evaluation/test_ai_gto_analyzer.py",
        ),
        dependencies=("training-orchestration",),
        diagnostics=("checkpoint-discovery",),
        focus="Checkpoint discovery, remote-safe model writes, and loader fallback policy.",
        criticality="high",
    ),
    SubsystemSpec(
        name="monitoring-observability",
        code_paths=(
            "src/poker_ai/logging_utils.py",
            "src/poker_ai/monitoring",
            "src/poker_ai/systems/logging.py",
            "tools/run_diagnostics.py",
        ),
        test_targets=(
            "tests/test_logging.py",
            "tests/test_logging_subsystem.py",
            "tests/test_logging_utils.py",
            "tests/monitoring",
            "tests/tools/test_run_diagnostics.py",
        ),
        diagnostics=("logging", "diagnostics"),
        focus="Structured logging, health hooks, and rapid diagnostics.",
        criticality="medium",
    ),
    SubsystemSpec(
        name="modular-services",
        code_paths=(
            "src/gatc_modular",
            "src/poker_ai/systems",
        ),
        test_targets=(
            "tests/modular",
            "tests/test_simple.py",
            "tests/test_systems_components.py",
            "tests/test_systems_structure.py",
        ),
        dependencies=("rules-engine", "cfr-algorithms", "self-play-runtime"),
        diagnostics=("subsystem-manifest",),
        focus="Explicit subsystem boundaries and dependency-aware composition.",
        criticality="medium",
    ),
)


def subsystem_manifest(
    specs: Sequence[SubsystemSpec] = DEFAULT_SUBSYSTEM_SPECS,
) -> list[dict[str, Any]]:
    return [spec.to_manifest() for spec in specs]


def select_impacted_subsystems(
    changed_paths: Iterable[str],
    specs: Sequence[SubsystemSpec] = DEFAULT_SUBSYSTEM_SPECS,
) -> tuple[list[SubsystemSpec], list[str]]:
    impacted: list[SubsystemSpec] = []
    unmatched: list[str] = []

    for path in changed_paths:
        matches = [spec for spec in specs if spec.owns_path(path)]
        if matches:
            for spec in matches:
                if spec not in impacted:
                    impacted.append(spec)
        else:
            unmatched.append(_normalise_path(path))

    impacted.sort(key=lambda spec: spec.name)
    unmatched.sort()
    return impacted, unmatched


def recommend_validation_plan(
    changed_paths: Iterable[str],
    specs: Sequence[SubsystemSpec] = DEFAULT_SUBSYSTEM_SPECS,
) -> dict[str, Any]:
    normalised_paths = [_normalise_path(path) for path in changed_paths]
    impacted, unmatched = select_impacted_subsystems(normalised_paths, specs=specs)

    test_targets = sorted({target for spec in impacted for target in spec.test_targets})
    diagnostics = sorted({mode for spec in impacted for mode in spec.diagnostics})

    return {
        "changed_paths": normalised_paths,
        "impacted_subsystems": [spec.to_manifest() for spec in impacted],
        "test_targets": test_targets,
        "diagnostics": diagnostics,
        "unmatched_paths": unmatched,
    }
