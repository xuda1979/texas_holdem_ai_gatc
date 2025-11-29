
import pytest
import torch
import shutil
import os
from poker_ai.ai.trainers.deep_cfr_trainer import DeepCFRTrainer
from poker_ai.selfplay.self_play import SelfPlay
from poker_ai.engine.texas_holdem import TexasHoldem

class TestDeepCFRIntegration:
    def setup_method(self):
        self.tmp_dir = "tests/deepcfr/tmp_deep_cfr"
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)

    def teardown_method(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_deep_cfr_training_loop(self):
        # Initialize trainer
        trainer = DeepCFRTrainer(
            input_feature_dim=18,
            hidden_dim=64,
            num_actions=10,
            learning_rate=1e-3,
            replay_buffer_capacity=1000,
            device="cpu"
        )

        # Initialize SelfPlay
        game_config = {
            "min_players": 2,
            "max_players": 2,
            "starting_stack": 1000,
            "big_blind": 20,
            "small_blind": 10
        }
        training_config = {
            "min_buffer_before_train": 10,
            "train_during_generation": False
        }

        self_play = SelfPlay(
            cfr_trainer=trainer,
            game_engine_config=game_config,
            training_config=training_config
        )

        # Generate some data
        # play_hand_for_training calls add_experience AND add_strategy_experience (patched)
        for i in range(5):
            self_play.play_hand_for_training(iteration=i+1)

        # Check buffers
        assert len(trainer.replay_buffer) > 0, "Advantage buffer should not be empty"
        assert len(trainer.strategy_buffer) > 0, "Strategy buffer should not be empty"

        # Train
        loss = trainer.train(batch_size=4)
        policy_loss = trainer.train_policy(batch_size=4)

        assert isinstance(loss, float)
        assert isinstance(policy_loss, float)

        # Test Save/Load
        save_path = os.path.join(self.tmp_dir, "model.pth")
        trainer.save_model(save_path)

        assert os.path.exists(save_path)

        # Create new trainer and load
        trainer2 = DeepCFRTrainer(
            input_feature_dim=18,
            hidden_dim=64,
            num_actions=10,
            learning_rate=1e-3,
            device="cpu"
        )
        trainer2.load_model(save_path)

        # Check if policy net loaded
        # We can check if weights match
        for p1, p2 in zip(trainer.policy_net.parameters(), trainer2.policy_net.parameters()):
            assert torch.allclose(p1, p2), "Policy network weights not restored correctly"
