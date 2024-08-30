import unittest
import torch
from ai_models.cfr import calculate_strategy, update_regret, update_strategy, compute_regrets

class TestCFR(unittest.TestCase):
    
    def test_calculate_strategy(self):
        cumulative_regret = torch.tensor([1.0, 2.0, 3.0])
        num_actions = 3
        strategy = calculate_strategy(cumulative_regret, num_actions)
        self.assertAlmostEqual(torch.sum(strategy).item(), 1.0, places=4, msg="Strategy should sum to 1")

    def test_update_regret(self):
        cumulative_regret = torch.tensor([1.0, 2.0, 3.0])
        regrets = torch.tensor([0.5, -0.5, 1.0])
        updated_regret = update_regret(cumulative_regret, regrets)
        self.assertTrue(torch.equal(updated_regret, torch.tensor([1.5, 1.5, 4.0])), "Regret update should be correct")

    def test_update_strategy(self):
        cumulative_strategy = torch.tensor([1.0, 1.0, 1.0])
        current_strategy = torch.tensor([0.2, 0.3, 0.5])
        updated_strategy = update_strategy(cumulative_strategy, current_strategy)
        self.assertTrue(torch.equal(updated_strategy, torch.tensor([1.2, 1.3, 1.5])), "Strategy update should be correct")

    def test_compute_regrets(self):
        payoffs = torch.tensor([2.0, 3.0, 4.0])
        action_values = torch.tensor([3.0, 3.0, 3.0])
        actual_action = 1
        regrets = compute_regrets(payoffs, action_values, actual_action)
        self.assertTrue(torch.equal(regrets, torch.tensor([-1.0, 0.0, 1.0])), "Regrets should be computed correctly")

if __name__ == "__main__":
    unittest.main()
