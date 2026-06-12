"""End-to-end convergence check of the Deep CFR stack on Kuhn poker.

The tabular CFR tests prove the regret-matching math; this test proves the
*deep* pipeline glue: external-sampling traversal targets, the advantage
replay buffer, strategy memory, linear-CFR weighted training and the policy
network all working together must drive exploitability well below the
uniform-random baseline on a game where the exact best response is computable.

A full convergence to Nash is not expected from a tiny network in seconds;
what must hold is a large, reproducible drop in exploitability.  A bug in any
stage (biased regret targets, wrong buffer wiring, broken loss weighting,
dead policy net) reliably destroys this margin.
"""

from __future__ import annotations

import itertools
import random

import pytest
import torch

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer
from poker_ai.games.kuhn import ACTIONS, best_response_value

CARDS = [1, 2, 3]
HIST_DIM = 4  # [is_start, action_pass, action_bet, position]
SEQ_LEN = 4
CARD_DIM = 17  # matches DeepCFRTrainer.card_feature_dim
INFOSET_HISTORIES = ["", "p", "b", "pb"]


def _encode(card: int, history: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a Kuhn infoset into the trainer's (hole, community, seq) format."""
    hole = torch.zeros(CARD_DIM)
    hole[card - 1] = 1.0
    community = torch.zeros(CARD_DIM)
    seq = torch.zeros(SEQ_LEN, HIST_DIM)
    seq[0, 0] = 1.0
    for i, action in enumerate(history):
        seq[i + 1, 1 if action == "p" else 2] = 1.0
        seq[i + 1, 3] = (i + 1) / SEQ_LEN
    return hole, community, seq


def _terminal_payoff(cards: list[int], history: str, player: int) -> float | None:
    if history == "pp":
        winner, pay = (0 if cards[0] > cards[1] else 1), 1
    elif history == "bp":
        winner, pay = 0, 1
    elif history == "pbp":
        winner, pay = 1, 1
    elif len(history) >= 2 and history.endswith("bb"):
        winner, pay = (0 if cards[0] > cards[1] else 1), 2
    else:
        return None
    return float(pay if winner == player else -pay)


def _regret_matched_policy(trainer: DeepCFRTrainer, card: int, history: str) -> torch.Tensor:
    hole, community, seq = _encode(card, history)
    advantages = trainer.get_advantages(hole, community, seq)
    positive = torch.clamp(advantages, min=0.0)
    total = positive.sum()
    if total.item() > 0:
        return positive / total
    return torch.ones(len(ACTIONS)) / len(ACTIONS)


def _traverse(
    trainer: DeepCFRTrainer,
    cards: list[int],
    history: str,
    traverser: int,
    iteration: int,
) -> float:
    """External Sampling MCCFR traversal feeding the Deep CFR memories."""
    payoff = _terminal_payoff(cards, history, traverser)
    if payoff is not None:
        return payoff

    current = len(history) % 2
    policy = _regret_matched_policy(trainer, cards[current], history)

    if current == traverser:
        utilities = torch.zeros(len(ACTIONS))
        for i, action in enumerate(ACTIONS):
            utilities[i] = _traverse(trainer, cards, history + action, traverser, iteration)
        node_value = float((utilities * policy).sum())
        hole, community, seq = _encode(cards[current], history)
        trainer.add_experience(
            hole, community, seq, regrets=utilities - node_value, iteration=iteration
        )
        return node_value

    # Opponent node: record average-strategy sample, then sample one action.
    hole, community, seq = _encode(cards[current], history)
    trainer.add_strategy_experience(
        hole, community, seq, strategy=policy.clone(), iteration=iteration
    )
    action_idx = int(torch.multinomial(policy, 1).item())
    return _traverse(trainer, cards, history + ACTIONS[action_idx], traverser, iteration)


def _extract_policy_net_strategy(trainer: DeepCFRTrainer) -> dict[str, list[float]]:
    strategy: dict[str, list[float]] = {}
    for card in CARDS:
        for history in INFOSET_HISTORIES:
            hole, community, seq = _encode(card, history)
            probs = trainer.get_policy(hole, community, seq)
            strategy[f"{card}{history}"] = [float(p) for p in probs]
    return strategy


def _exploitability(strategy: dict[str, list[float]]) -> float:
    return (best_response_value(strategy, 0) + best_response_value(strategy, 1)) / 2.0


@pytest.fixture()
def shallow_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a 1-layer transformer so the toy run finishes in seconds."""
    monkeypatch.setattr(AdvantageNetwork, "DEFAULT_NUM_LAYERS", 1)


def test_deep_cfr_pipeline_reduces_exploitability_on_kuhn(shallow_network: None) -> None:
    random.seed(1)
    torch.manual_seed(1)

    trainer = DeepCFRTrainer(
        input_feature_dim=HIST_DIM,
        hidden_dim=32,
        num_actions=len(ACTIONS),
        learning_rate=2e-3,
        replay_buffer_capacity=20_000,
        device="cpu",
    )
    # Deterministic data generation: disable dropout noise in the policies.
    trainer.advantage_net.eval()
    trainer.policy_net.eval()

    uniform = {
        f"{card}{history}": [0.5, 0.5] for card in CARDS for history in INFOSET_HISTORIES
    }
    eps_uniform = _exploitability(uniform)
    assert eps_uniform == pytest.approx(0.4583, abs=1e-3)  # sanity: known baseline

    deals = list(itertools.permutations(CARDS, 2))
    for iteration in range(1, 61):
        for traverser in (0, 1):
            for deal in deals:
                _traverse(trainer, list(deal), "", traverser, iteration)
        for _ in range(4):
            trainer.train(batch_size=256)
            trainer.train_policy(batch_size=256)

    assert len(trainer.replay_buffer) > 0
    assert len(trainer.strategy_buffer) > 0

    eps_final = _exploitability(_extract_policy_net_strategy(trainer))
    # The learned average policy must beat the uniform baseline by a wide,
    # bug-detecting margin (measured ~0.13-0.16 across seeds; uniform 0.458).
    assert eps_final < 0.6 * eps_uniform
    assert eps_final < 0.3
