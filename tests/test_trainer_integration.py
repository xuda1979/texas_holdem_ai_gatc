import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer


def test_trainer_updates_info_set():
    trainer = AICFRTrainer(device="cpu")
    card_dim = trainer.model.card_projection.in_features
    history_dim = trainer.model.history_projection.in_features
    hole = torch.zeros(card_dim)
    community = torch.zeros(card_dim)
    history = torch.zeros(1, history_dim)
    payoffs = torch.arange(trainer.num_actions, dtype=torch.float32)
    info_set = "root"
    trainer.train(info_set, hole, community, history, payoffs)
    assert info_set in trainer.cumulative_regret
    assert torch.any(trainer.cumulative_regret[info_set] != 0)
    assert info_set in trainer.cumulative_strategy
    strategy = trainer.get_final_average_strategy(info_set)
    assert torch.isclose(strategy.sum(), torch.tensor(1.0), atol=1e-6)
