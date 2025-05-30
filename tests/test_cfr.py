import unittest
import torch
from rules.cfr import (
    calculate_strategy,
    update_regret,
    update_strategy,
    compute_regrets,
    compute_average_strategy,
    cfr_iteration
)

# MockGame class for testing cfr_iteration
class MockGame:
    """
    A mock game environment for testing cfr_iteration.
    It returns predefined counterfactual payoffs for actions.
    """
    def __init__(self, counterfactual_payoffs_map: dict):
        """
        Args:
            counterfactual_payoffs_map (dict): A dictionary mapping action_idx to its counterfactual payoff.
                                               Example: {0: 10.0, 1: -5.0, 2: 20.0}
        """
        self.counterfactual_payoffs_map = counterfactual_payoffs_map

    def simulate_action(self, action_idx: int) -> float:
        """
        Simulates taking an action and returns its counterfactual payoff.
        Args:
            action_idx (int): The index of the action taken.
        Returns:
            float: The predefined counterfactual payoff for the action.
        """
        return self.counterfactual_payoffs_map.get(action_idx, 0.0)


class TestCFR(unittest.TestCase):
    
    def test_calculate_strategy(self):
        # Case 1: All regrets are positive
        cumulative_regret1 = torch.tensor([1.0, 2.0, 3.0]) # Sum = 6
        num_actions1 = 3
        strategy1 = calculate_strategy(cumulative_regret1, num_actions1)
        self.assertTrue(torch.allclose(strategy1, torch.tensor([1/6, 2/6, 3/6])), "Strategy should be proportional to positive regrets")
        self.assertAlmostEqual(torch.sum(strategy1).item(), 1.0, places=6, msg="Strategy should sum to 1")

        # Case 2: Mixed positive and negative regrets
        cumulative_regret2 = torch.tensor([-1.0, 0.0, 3.0, 1.0]) # Positive part: [0,0,3,1], Sum = 4
        num_actions2 = 4
        strategy2 = calculate_strategy(cumulative_regret2, num_actions2)
        self.assertTrue(torch.allclose(strategy2, torch.tensor([0.0, 0.0, 3/4, 1/4])), "Strategy should ignore negative regrets")
        self.assertAlmostEqual(torch.sum(strategy2).item(), 1.0, places=6, msg="Strategy should sum to 1")

        # Case 3: All regrets are non-positive (sum_positive_regret == 0)
        cumulative_regret3 = torch.tensor([-1.0, -2.0, 0.0])
        num_actions3 = 3
        strategy3 = calculate_strategy(cumulative_regret3, num_actions3)
        self.assertTrue(torch.allclose(strategy3, torch.tensor([1/3, 1/3, 1/3])), "Should return uniform strategy if all regrets are non-positive")
        self.assertAlmostEqual(torch.sum(strategy3).item(), 1.0, places=6, msg="Uniform strategy should sum to 1")

        # Case 4: Single action
        cumulative_regret4 = torch.tensor([10.0])
        num_actions4 = 1
        strategy4 = calculate_strategy(cumulative_regret4, num_actions4)
        self.assertTrue(torch.allclose(strategy4, torch.tensor([1.0])), "Strategy for single action should be 1.0")
        self.assertAlmostEqual(torch.sum(strategy4).item(), 1.0, places=6, msg="Strategy should sum to 1")

    def test_update_regret(self):
        cumulative_regret = torch.tensor([1.0, 2.0, 3.0])
        regrets = torch.tensor([0.5, -0.5, 1.0])
        updated_regret = update_regret(cumulative_regret, regrets)
        self.assertTrue(torch.equal(updated_regret, torch.tensor([1.5, 1.5, 4.0])), "Regret update should be element-wise addition")

        cumulative_regret_empty = torch.empty(0)
        regrets_empty = torch.empty(0)
        updated_regret_empty = update_regret(cumulative_regret_empty, regrets_empty)
        self.assertTrue(torch.equal(updated_regret_empty, torch.empty(0)), "Update with empty tensors should work")


    def test_update_strategy(self):
        cumulative_strategy = torch.tensor([1.0, 1.0, 1.0])
        current_strategy = torch.tensor([0.2, 0.3, 0.5])
        updated_strategy = update_strategy(cumulative_strategy, current_strategy)
        self.assertTrue(torch.equal(updated_strategy, torch.tensor([1.2, 1.3, 1.5])), "Strategy update should be element-wise addition")

        cumulative_strategy_empty = torch.empty(0)
        current_strategy_empty = torch.empty(0)
        updated_strategy_empty = update_strategy(cumulative_strategy_empty, current_strategy_empty)
        self.assertTrue(torch.equal(updated_strategy_empty, torch.empty(0)), "Update with empty tensors should work")

    def test_compute_regrets(self):
        # Test case 1: Basic scenario
        action_counterfactual_values1 = torch.tensor([10.0, 12.0, 8.0])
        state_value1 = 10.0 # If current strategy led to this state value
        expected_regrets1 = torch.tensor([0.0, 2.0, -2.0])
        regrets1 = compute_regrets(action_counterfactual_values1, state_value1)
        self.assertTrue(torch.allclose(regrets1, expected_regrets1), "Regrets should be action_values - state_value")

        # Test case 2: All action values are same as state value
        action_counterfactual_values2 = torch.tensor([5.0, 5.0, 5.0])
        state_value2 = 5.0
        expected_regrets2 = torch.tensor([0.0, 0.0, 0.0])
        regrets2 = compute_regrets(action_counterfactual_values2, state_value2)
        self.assertTrue(torch.allclose(regrets2, expected_regrets2), "Regrets should be zero if all action values equal state value")
        
        # Test case 3: Scalar state value, vector action values
        action_counterfactual_values3 = torch.tensor([1.0, -2.0, 3.0])
        state_value3 = 0.5
        expected_regrets3 = torch.tensor([0.5, -2.5, 2.5])
        regrets3 = compute_regrets(action_counterfactual_values3, state_value3)
        self.assertTrue(torch.allclose(regrets3, expected_regrets3), "Regrets computation with scalar state value")


    def test_compute_average_strategy(self):
        # Case 1: Non-zero sum_strategy
        cumulative_strategy1 = torch.tensor([1.0, 2.0, 3.0]) # Sum = 6
        avg_strategy1 = compute_average_strategy(cumulative_strategy1)
        self.assertTrue(torch.allclose(avg_strategy1, torch.tensor([1/6, 2/6, 3/6])), "Average strategy incorrect")
        self.assertAlmostEqual(torch.sum(avg_strategy1).item(), 1.0, places=6, msg="Average strategy should sum to 1")

        # Case 2: sum_strategy == 0
        num_actions2 = 4
        cumulative_strategy2 = torch.zeros(num_actions2)
        avg_strategy2 = compute_average_strategy(cumulative_strategy2)
        self.assertTrue(torch.allclose(avg_strategy2, torch.ones(num_actions2) / num_actions2), "Should return uniform strategy if sum is zero")
        self.assertAlmostEqual(torch.sum(avg_strategy2).item(), 1.0, places=6, msg="Uniform average strategy should sum to 1")

        # Case 3: Single action
        cumulative_strategy3 = torch.tensor([10.0])
        avg_strategy3 = compute_average_strategy(cumulative_strategy3)
        self.assertTrue(torch.allclose(avg_strategy3, torch.tensor([1.0])), "Average strategy for single action should be 1.0")


    def test_cfr_iteration_single_iteration(self):
        num_actions = 3
        # Iteration 1:
        # Initial cumulative_regret = [0,0,0] -> current_strategy = [1/3, 1/3, 1/3]
        # Mock payoffs: action0=0.6, action1=0.3, action2=0.0
        # action_counterfactual_values = [0.6, 0.3, 0.0]
        # state_value = (1/3)*0.6 + (1/3)*0.3 + (1/3)*0.0 = 0.2 + 0.1 + 0.0 = 0.3
        # regrets = [0.6-0.3, 0.3-0.3, 0.0-0.3] = [0.3, 0.0, -0.3]
        # cumulative_regret_after_1_iter = [0.3, 0.0, -0.3]
        # cumulative_strategy_after_1_iter = [1/3, 1/3, 1/3]

        mock_game = MockGame({0: 0.6, 1: 0.3, 2: 0.0})
        cumulative_regret = torch.zeros(num_actions)
        cumulative_strategy = torch.zeros(num_actions)
        
        final_regret, final_strategy = cfr_iteration(
            game=mock_game,
            cumulative_regret=cumulative_regret,
            cumulative_strategy=cumulative_strategy,
            num_actions=num_actions,
            num_iterations=1
        )

        expected_final_regret = torch.tensor([0.3, 0.0, -0.3])
        expected_final_strategy = torch.tensor([1/3, 1/3, 1/3])

        self.assertEqual(final_regret.shape, (num_actions,))
        self.assertEqual(final_strategy.shape, (num_actions,))
        self.assertTrue(torch.allclose(final_regret, expected_final_regret, atol=1e-6), 
                        f"Regrets mismatch. Expected {expected_final_regret}, Got {final_regret}")
        self.assertTrue(torch.allclose(final_strategy, expected_final_strategy, atol=1e-6),
                        f"Strategy mismatch. Expected {expected_final_strategy}, Got {final_strategy}")

    def test_cfr_iteration_two_iterations(self):
        num_actions = 2
        # Payoffs: action0=1.0, action1=0.0
        mock_game = MockGame({0: 1.0, 1: 0.0})
        
        cumulative_regret = torch.zeros(num_actions)
        cumulative_strategy = torch.zeros(num_actions)

        # Iteration 1:
        # Initial cumulative_regret = [0,0] -> current_strategy_1 = [0.5, 0.5]
        # action_counterfactual_values = [1.0, 0.0]
        # state_value_1 = 0.5*1.0 + 0.5*0.0 = 0.5
        # regrets_1 = [1.0-0.5, 0.0-0.5] = [0.5, -0.5]
        # cumulative_regret_1 = [0.5, -0.5]
        # cumulative_strategy_1 = [0.5, 0.5] (current_strategy_1 accumulated)

        # Iteration 2:
        # cumulative_regret_1 = [0.5, -0.5] -> positive part = [0.5, 0] sum=0.5 -> current_strategy_2 = [1.0, 0.0]
        # action_counterfactual_values = [1.0, 0.0] (same payoffs from mock game)
        # state_value_2 = 1.0*1.0 + 0.0*0.0 = 1.0
        # regrets_2 = [1.0-1.0, 0.0-1.0] = [0.0, -1.0]
        # cumulative_regret_2 = [0.5+0.0, -0.5-1.0] = [0.5, -1.5]
        # cumulative_strategy_2 = [0.5, 0.5] (from iter1) + [1.0, 0.0] (from iter2) = [1.5, 0.5]

        final_regret, final_strategy = cfr_iteration(
            game=mock_game,
            cumulative_regret=cumulative_regret,
            cumulative_strategy=cumulative_strategy,
            num_actions=num_actions,
            num_iterations=2
        )

        expected_final_regret = torch.tensor([0.5, -1.5])
        expected_final_strategy = torch.tensor([1.5, 0.5])
        
        self.assertTrue(torch.allclose(final_regret, expected_final_regret, atol=1e-6),
                        f"Regrets mismatch. Expected {expected_final_regret}, Got {final_regret}")
        self.assertTrue(torch.allclose(final_strategy, expected_final_strategy, atol=1e-6),
                        f"Strategy mismatch. Expected {expected_final_strategy}, Got {final_strategy}")

if __name__ == "__main__":
    unittest.main()
