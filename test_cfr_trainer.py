import unittest
import numpy as np
from cfr_trainer import CFRTrainer
from texas_holdem import TexasHoldemGame

class TestCFRTrainer(unittest.TestCase):

    def setUp(self):
        self.config = {
            'input_shape': (10, 10, 1),  # Example shape, update as per your game state encoding
            'num_actions': 10,
            'learning_rate': 0.001,
            'num_res_blocks': 3,
            'num_players': 2
        }
        self.trainer = CFRTrainer(self.config)

    def test_model_building(self):
        self.assertIsNotNone(self.trainer.model, "Model should be built and not None")

    def test_train_step(self):
        states = np.random.rand(5, *self.config['input_shape'])
        regrets = np.random.rand(5, self.config['num_actions'])
        loss = self.trainer.train_step(states, regrets)
        self.assertGreater(loss, 0, "Loss should be greater than 0")

    def test_cfr(self):
        game = TexasHoldemGame(self.config['num_players'])
        initial_state = game.get_initial_state()
        utility = self.trainer.cfr(initial_state, player=0, iteration=1)
        self.assertIsNotNone(utility, "Utility should be calculated")

    def test_strategy_computation(self):
        state_representation = np.random.rand(*self.config['input_shape'])
        strategy = self.trainer.get_strategy(state_representation)
        self.assertAlmostEqual(sum(strategy), 1.0, "Strategy probabilities should sum to 1")

if __name__ == '__main__':
    unittest.main()
