import torch
from trainers import ai_cfr_trainer


def test_training_logs_loss(tmp_path):
    log_file = tmp_path / "train.log"
    model_path = tmp_path / "model.pth"
    config = {
        'logging': {'log_file': str(log_file)},
        'model': {'hidden_dim': 8, 'num_actions': 2, 'learning_rate': 0.001, 'd_raw_feature': 3},
        'training': {'save_model_path': str(model_path)}
    }

    ai_cfr_trainer.config = config
    logger = ai_cfr_trainer.logging.getLogger()
    logger.setLevel(ai_cfr_trainer.logging.INFO)
    handler = ai_cfr_trainer.logging.FileHandler(log_file)
    handler.setLevel(ai_cfr_trainer.logging.INFO)
    logger.addHandler(handler)

    trainer = ai_cfr_trainer.AICFRTrainer(device="cpu")

    state = torch.zeros(5, 3)
    payoffs = torch.tensor([1.0, -1.0])
    trainer.train(state, payoffs)

    handler.flush()
    logger.removeHandler(handler)
    handler.close()

    contents = log_file.read_text()
    assert "Training step completed. Loss:" in contents
