"""Model health checks for AdvantageNetwork-based poker strategies.

This module provides a suite of *non-gameplay* diagnostics that look directly
at the network weights and the per-state action distributions it produces.
These checks are designed to catch training bugs early -- before a broken
checkpoint has time to pollute the model pool -- and to give developers a
quick "is this checkpoint even sane?" answer when inspecting snapshots.

The checks are deliberately *cheap* (no full matches are played) so that they
can be invoked from inside the training loop or against a directory full of
checkpoints in a few seconds.

Categories
----------
1. Weight checks -- look for NaN/Inf, dead units, exploding magnitudes.
2. Policy checks -- sample a handful of random game states and inspect the
   implied regret-matched policy for entropy, legal-action mass, and
   consistency across repeated calls.
3. Determinism checks -- ensure the same state yields the same advantages
   (modulo sampling noise) under ``torch.no_grad``/eval mode.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.engine.texas_holdem import TexasHoldem
from poker_ai.utils.action_mapping import get_legal_actions_mask
from poker_ai.utils.state_representation import (
    infer_normalization_scale,
    prepare_transformer_input,
)


@dataclass
class HealthReport:
    """Container for the result of a single checkpoint health check."""

    checkpoint_path: str
    ok: bool
    severity: str  # "ok" | "warning" | "error"
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.severity == "ok"

    @property
    def has_errors(self) -> bool:
        return any(c.get("severity") == "error" for c in self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"HealthReport({self.checkpoint_path}) severity={self.severity} ok={self.ok}",
        ]
        for name, check in self.checks.items():
            lines.append(
                f"  - {name}: severity={check.get('severity')} "
                f"value={check.get('value')!r} msg={check.get('message')}"
            )
        for key, value in self.diagnostics.items():
            lines.append(f"  + {key}: {value}")
        return "\n".join(lines)


def _set_check(
    report: HealthReport,
    name: str,
    *,
    severity: str,
    value: Any,
    message: str,
) -> None:
    report.checks[name] = {
        "severity": severity,
        "value": value,
        "message": message,
    }
    if severity == "error":
        report.severity = "error"
        report.ok = False
    elif severity == "warning" and report.severity != "error":
        report.severity = "warning"


def _iter_named_parameters(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        yield name, param.detach()


def check_weight_health(model: AdvantageNetwork, report: HealthReport) -> None:
    """Inspect raw parameter tensors for NaN/Inf, dead units, and scale issues."""

    num_nan = 0
    num_inf = 0
    max_abs = 0.0
    min_abs_finite = float("inf")
    total_params = 0
    dead_params = 0  # parameters whose values are all zero
    exploding_params: list[str] = []

    for name, tensor in _iter_named_parameters(model):
        total_params += tensor.numel()
        if torch.isnan(tensor).any().item():
            num_nan += int(torch.isnan(tensor).sum().item())
            _set_check(
                report,
                f"nan_weights/{name}",
                severity="error",
                value=True,
                message=f"Parameter {name} contains NaN values",
            )
        if torch.isinf(tensor).any().item():
            num_inf += int(torch.isinf(tensor).sum().item())
            _set_check(
                report,
                f"inf_weights/{name}",
                severity="error",
                value=True,
                message=f"Parameter {name} contains Inf values",
            )
        finite = torch.isfinite(tensor)
        if finite.any().item():
            tmax = float(tensor[finite].abs().max().item())
            tmin = float(tensor[finite].abs().min().item())
            max_abs = max(max_abs, tmax)
            min_abs_finite = min(min_abs_finite, tmin)
            if tmax > 1e4:
                exploding_params.append(f"{name}:{tmax:.2e}")
            if not tensor.abs().any().item():
                dead_params += 1
        else:
            dead_params += 1

    _set_check(
        report,
        "weight_nan_count",
        severity="error" if num_nan > 0 else "ok",
        value=num_nan,
        message=f"{num_nan} NaN parameter values detected",
    )
    _set_check(
        report,
        "weight_inf_count",
        severity="error" if num_inf > 0 else "ok",
        value=num_inf,
        message=f"{num_inf} Inf parameter values detected",
    )
    _set_check(
        report,
        "weight_max_abs",
        severity="error" if max_abs > 1e4 else ("warning" if max_abs > 1e3 else "ok"),
        value=max_abs,
        message=f"Largest |weight|={max_abs:.4e}",
    )
    _set_check(
        report,
        "weight_dead_params",
        severity="warning" if dead_params > 0 else "ok",
        value=dead_params,
        message=f"{dead_params} all-zero parameter tensors",
    )
    if exploding_params:
        _set_check(
            report,
            "weight_exploding",
            severity="error",
            value=exploding_params[:5],
            message=f"{len(exploding_params)} parameters exceed |w|>1e4",
        )

    report.diagnostics["weight_total_params"] = total_params
    report.diagnostics["weight_min_abs_finite"] = (
        min_abs_finite if math.isfinite(min_abs_finite) else None
    )


def _sample_random_states(
    num_states: int,
    *,
    max_seq_len: int,
    d_raw_feature: int,
    num_players: int = 2,
    starting_stack: int = 1000,
    big_blind: int = 10,
    small_blind: int = 5,
    seed: int = 0,
) -> list[tuple[TexasHoldem, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate random mid-hand game states with their transformer inputs.

    Each sampled state is a fresh game advanced by a few random legal actions
    so that we cover a variety of betting situations (preflop, postflop, etc).
    """

    rng = random.Random(seed)
    samples: list[tuple] = []
    for i in range(num_states):
        game = TexasHoldem(
            num_players=num_players,
            starting_stack=starting_stack,
            verbose=False,
        )
        game.rules.big_blind = big_blind
        game.rules.small_blind = small_blind
        # Seed the deck RNG deterministically for reproducibility.
        try:
            game.rules.deck_manager.rng.seed(rng.randint(0, 2**31 - 1))
        except AttributeError:  # pragma: no cover - defensive
            pass
        game.initialize_game()
        # Take 0..4 random legal actions to land in a variety of states.
        num_actions = rng.randint(0, 4)
        for _ in range(num_actions):
            if game.is_hand_over():
                break
            player = game.rules.current_player
            valid = game.get_valid_actions(player)
            if not valid:
                break
            action = rng.choice(valid)
            amount: int | None = None
            if action in ("raise", "bet"):
                try:
                    lo = game.get_min_raise_amount(player)
                    hi = game.get_max_raise_amount(player)
                except Exception:
                    lo, hi = big_blind, 2 * big_blind
                if hi < lo:
                    action = "check" if "check" in valid else "call"
                else:
                    amount = rng.randint(lo, max(hi, lo))
            try:
                game.process_action(player, action, amount)
            except Exception:
                break
            try:
                game.rules.advance_turn()
            except Exception:  # pragma: no cover - defensive
                pass
        if game.is_hand_over():
            continue
        # Pick the player whose turn it is to act.
        player = game.rules.current_player
        try:
            scale_hint = float(big_blind)
            normalization_scale = infer_normalization_scale(game, scale_hint)
        except Exception:
            normalization_scale = float(big_blind)
        try:
            hole, community, history, mask = prepare_transformer_input(
                game,
                player,
                max_seq_len,
                d_raw_feature,
                normalization_scale=normalization_scale,
                return_mask=True,
            )
        except Exception:
            continue
        samples.append((game, player, hole, community, history, mask))
    return samples


def _regret_matched_policy(
    model: AdvantageNetwork,
    hole: torch.Tensor,
    community: torch.Tensor,
    history: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    device: torch.device,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the regret-matched policy implied by the advantage network."""

    with torch.no_grad():
        kwargs: dict[str, Any] = {}
        if key_padding_mask is not None:
            kwargs["key_padding_mask"] = (~key_padding_mask.unsqueeze(0)).to(device)
        adv = model(
            hole.unsqueeze(0).to(device),
            community.unsqueeze(0).to(device),
            history.unsqueeze(0).to(device),
            **kwargs,
        ).squeeze(0)
    positive = torch.clamp(adv, min=0.0)
    masked = torch.where(
        legal_mask.to(device).bool(),
        positive,
        torch.zeros_like(positive, device=device),
    )
    total = masked.sum()
    if total.item() > 0:
        return masked / total
    # If all positive regrets are zero, fall back to uniform over legal actions.
    return legal_mask.to(device).float() / max(int(legal_mask.sum().item()), 1)


def check_policy_health(
    model: AdvantageNetwork,
    report: HealthReport,
    *,
    num_states: int = 16,
    num_actions: int = 10,
    max_seq_len: int = 256,
    d_raw_feature: int = 18,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> None:
    """Sample random states and inspect the implied policy distribution."""

    device = torch.device(device)
    model.eval()
    samples = _sample_random_states(
        num_states=num_states,
        max_seq_len=max_seq_len,
        d_raw_feature=d_raw_feature,
        seed=seed,
    )

    if not samples:
        _set_check(
            report,
            "policy_sampling",
            severity="warning",
            value=0,
            message="Could not sample any valid game states for policy inspection",
        )
        return

    entropies: list[float] = []
    legal_masses: list[float] = []
    nan_logits_count = 0
    inf_logits_count = 0
    illegal_argmax_count = 0
    determinism_mismatches = 0
    max_policy_values: list[float] = []
    min_nonzero_policy_values: list[float] = []

    for game, player, hole, community, history, _mask in samples:
        legal_mask = get_legal_actions_mask(game, player, num_actions)
        with torch.no_grad():
            adv = model(
                hole.unsqueeze(0).to(device),
                community.unsqueeze(0).to(device),
                history.unsqueeze(0).to(device),
            ).squeeze(0)
        if torch.isnan(adv).any().item():
            nan_logits_count += 1
            continue
        if torch.isinf(adv).any().item():
            inf_logits_count += 1
            continue

        # Determinism: re-run and compare.
        with torch.no_grad():
            adv2 = model(
                hole.unsqueeze(0).to(device),
                community.unsqueeze(0).to(device),
                history.unsqueeze(0).to(device),
            ).squeeze(0)
        if not torch.allclose(adv, adv2, atol=1e-6, equal_nan=True):
            determinism_mismatches += 1

        policy = _regret_matched_policy(
            model, hole, community, history, legal_mask, device=device
        )
        if torch.isnan(policy).any().item() or torch.isinf(policy).any().item():
            nan_logits_count += 1
            continue

        # Entropy of the policy in nats.
        p = torch.clamp(policy, min=1e-12)
        entropy = float(-(p * torch.log(p)).sum().item())
        entropies.append(entropy)

        # Mass on legal actions (should be 1.0; anything else is a bug).
        legal_mass = float(policy[legal_mask.to(device).bool()].sum().item())
        legal_masses.append(legal_mass)

        max_policy_values.append(float(policy.max().item()))
        nonzero = policy[policy > 0]
        if nonzero.numel() > 0:
            min_nonzero_policy_values.append(float(nonzero.min().item()))

        # Argmax must be a legal action.
        argmax = int(policy.argmax().item())
        if not bool(legal_mask[argmax].item()):
            illegal_argmax_count += 1

    if nan_logits_count or inf_logits_count:
        _set_check(
            report,
            "policy_nan_inf",
            severity="error",
            value=nan_logits_count + inf_logits_count,
            message=f"{nan_logits_count} NaN and {inf_logits_count} Inf advantage outputs detected",
        )
    else:
        _set_check(
            report,
            "policy_nan_inf",
            severity="ok",
            value=0,
            message="No NaN/Inf advantage outputs across sampled states",
        )

    if determinism_mismatches:
        _set_check(
            report,
            "policy_determinism",
            severity="error",
            value=determinism_mismatches,
            message=f"{determinism_mismatches} states produced different advantages on re-run",
        )
    else:
        _set_check(
            report,
            "policy_determinism",
            severity="ok",
            value=0,
            message="Advantage outputs are deterministic in eval mode",
        )

    if illegal_argmax_count:
        _set_check(
            report,
            "policy_argmax_illegal",
            severity="error",
            value=illegal_argmax_count,
            message=f"{illegal_argmax_count} states had argmax on an illegal action",
        )
    else:
        _set_check(
            report,
            "policy_argmax_illegal",
            severity="ok",
            value=0,
            message="Argmax always falls on a legal action",
        )

    if entropies:
        mean_entropy = sum(entropies) / len(entropies)
        # Heads-up Hold'em with ~3-10 legal actions: a healthy regret-matched
        # policy typically has entropy in [0.2, 2.2] nats.  Outside that band
        # is suspicious but not necessarily wrong, so we warn.
        if mean_entropy < 0.05:
            sev = "warning"
            msg = f"Mean policy entropy {mean_entropy:.4f} is suspiciously low (near-deterministic)"
        elif mean_entropy > 2.6:
            sev = "warning"
            msg = f"Mean policy entropy {mean_entropy:.4f} is suspiciously high (near-uniform)"
        else:
            sev = "ok"
            msg = f"Mean policy entropy {mean_entropy:.4f} nats"
        _set_check(report, "policy_mean_entropy", severity=sev, value=mean_entropy, message=msg)
        report.diagnostics["policy_entropy_min"] = min(entropies)
        report.diagnostics["policy_entropy_max"] = max(entropies)
        report.diagnostics["policy_entropy_samples"] = len(entropies)

    if legal_masses:
        mean_legal_mass = sum(legal_masses) / len(legal_masses)
        if abs(mean_legal_mass - 1.0) > 1e-3:
            _set_check(
                report,
                "policy_legal_mass",
                severity="error",
                value=mean_legal_mass,
                message=f"Policy mass on legal actions is {mean_legal_mass:.4f} (must be 1.0)",
            )
        else:
            _set_check(
                report,
                "policy_legal_mass",
                severity="ok",
                value=mean_legal_mass,
                message="Policy mass on legal actions is 1.0",
            )

    if max_policy_values:
        report.diagnostics["policy_max_max"] = max(max_policy_values)
        report.diagnostics["policy_max_min"] = min(max_policy_values)
        # A policy that always puts ~100% mass on one action is "collapsed".
        collapsed = sum(1 for v in max_policy_values if v > 0.999)
        if collapsed == len(max_policy_values) and len(max_policy_values) >= 5:
            _set_check(
                report,
                "policy_collapsed",
                severity="warning",
                value=collapsed,
                message="Policy is degenerate (always puts ~100% mass on one action)",
            )
        else:
            _set_check(
                report,
                "policy_collapsed",
                severity="ok",
                value=collapsed,
                message=f"{collapsed}/{len(max_policy_values)} states had a collapsed policy",
            )


def check_metadata_consistency(
    metadata: dict[str, Any] | None,
    *,
    expected_num_actions: int | None = None,
    expected_hidden_dim: int | None = None,
) -> dict[str, Any]:
    """Validate the metadata block embedded in a checkpoint payload."""

    issues: list[str] = []
    if not metadata:
        issues.append("metadata is missing or empty")
        return {"ok": False, "issues": issues, "metadata": metadata}

    required = ("history_feature_dim", "card_feature_dim", "num_actions", "hidden_dim")
    for key in required:
        if key not in metadata:
            issues.append(f"missing required key: {key}")

    if expected_num_actions is not None and metadata.get("num_actions") != expected_num_actions:
        issues.append(
            f"num_actions={metadata.get('num_actions')} != expected={expected_num_actions}"
        )
    if expected_hidden_dim is not None and metadata.get("hidden_dim") != expected_hidden_dim:
        issues.append(
            f"hidden_dim={metadata.get('hidden_dim')} != expected={expected_hidden_dim}"
        )

    return {"ok": not issues, "issues": issues, "metadata": dict(metadata)}


def run_health_check(
    model: AdvantageNetwork,
    *,
    checkpoint_path: str = "<in-memory>",
    metadata: dict[str, Any] | None = None,
    num_states: int = 16,
    num_actions: int = 10,
    max_seq_len: int = 256,
    d_raw_feature: int = 18,
    seed: int = 0,
    device: torch.device | str = "cpu",
    expected_num_actions: int | None = None,
    expected_hidden_dim: int | None = None,
) -> HealthReport:
    """Run all health checks on a model and return a single report."""

    report = HealthReport(checkpoint_path=checkpoint_path, ok=True, severity="ok")
    check_weight_health(model, report)
    check_policy_health(
        model,
        report,
        num_states=num_states,
        num_actions=num_actions,
        max_seq_len=max_seq_len,
        d_raw_feature=d_raw_feature,
        seed=seed,
        device=device,
    )
    meta_check = check_metadata_consistency(
        metadata,
        expected_num_actions=expected_num_actions,
        expected_hidden_dim=expected_hidden_dim,
    )
    _set_check(
        report,
        "metadata",
        severity="ok" if meta_check["ok"] else "error",
        value=meta_check["ok"],
        message="; ".join(meta_check["issues"]) if meta_check["issues"] else "metadata valid",
    )
    if meta_check["metadata"]:
        report.diagnostics["metadata"] = meta_check["metadata"]

    if not report.has_errors and report.severity == "ok":
        report.ok = True
    return report


__all__ = [
    "HealthReport",
    "check_weight_health",
    "check_policy_health",
    "check_metadata_consistency",
    "run_health_check",
]
