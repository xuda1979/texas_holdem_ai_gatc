import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.ai import trainers as ai_cfr_trainer


def test_training_logs_loss(tmp_path):
    log_file = tmp_path / "train.log"
    model_path = tmp_path / "model.pth"
    config = {
        'logging': {'log_file': str(log_file)},
        'model': {'hidden_dim': 8, 'num_actions': 2, 'learning_rate': 0.001, 'd_raw_feature': 3},
        'training': {'save_model_path': str(model_path)}
    }

    original_config_pkg = ai_cfr_trainer.config
    original_config_module = ai_cfr_trainer.ai_cfr_trainer_module.config
    try:
        ai_cfr_trainer.config = config
        ai_cfr_trainer.ai_cfr_trainer_module.config = config

        logger = ai_cfr_trainer.logging.getLogger()
        logger.setLevel(ai_cfr_trainer.logging.INFO)
        handler = ai_cfr_trainer.logging.FileHandler(log_file)
        handler.setLevel(ai_cfr_trainer.logging.INFO)
        logger.addHandler(handler)

        trainer = ai_cfr_trainer.AICFRTrainer(device="cpu")

        card_dim = trainer.model.card_projection.in_features
        history_dim = trainer.model.history_projection.in_features
        hole = torch.zeros(card_dim)
        community = torch.zeros(card_dim)
        state = torch.zeros(5, history_dim)
        payoffs = torch.tensor([1.0, -1.0])
        trainer.train("root", hole, community, state, payoffs)

        handler.flush()
        logger.removeHandler(handler)
        handler.close()

        contents = log_file.read_text()
        assert "Training step completed. Loss:" in contents
    finally:
        ai_cfr_trainer.config = original_config_pkg
        ai_cfr_trainer.ai_cfr_trainer_module.config = original_config_module
