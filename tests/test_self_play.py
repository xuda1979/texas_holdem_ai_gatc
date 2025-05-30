import unittest
from unittest.mock import MagicMock, patch
import torch # Required for dummy tensors if cfr_trainer.train is called
import sys
import os

# Adjust the Python path to include the root directory of the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from self_play.self_play import SelfPlay
# from trainers.ai_cfr_trainer import AICFRTrainer # For type hinting if needed
# from game_engine.texas_holdem import TexasHoldem # For type hinting if needed

class TestSelfPlay(unittest.TestCase):

    def setUp(self):
        # Mock AICFRTrainer
        self.mock_cfr_trainer = MagicMock()

        # Mock Game Engine Config
        self.game_engine_config = {
            'num_players': 2,
            'starting_stack': 1000,
            'big_blind': 10,
            'small_blind': 5
            # Other parameters like 'player_strategies' are handled by SelfPlay's init
        }

    @patch('self_play.self_play.TexasHoldem') # Patch where TexasHoldem is looked up
    def test_init_successful(self, mock_texas_holdem_class):
        """Test successful initialization of SelfPlay."""
        mock_game_engine_instance = MagicMock()
        mock_texas_holdem_class.return_value = mock_game_engine_instance

        sp = SelfPlay(cfr_trainer=self.mock_cfr_trainer, game_engine_config=self.game_engine_config)

        self.assertEqual(sp.cfr_trainer, self.mock_cfr_trainer)
        self.assertEqual(sp.game_engine_config, self.game_engine_config)
        self.assertEqual(sp.game_engine, mock_game_engine_instance)

        # Check if TexasHoldem was instantiated with the correct config
        # and that player_strategies were set up (SelfPlay uses DummyStrategy by default)
        args, kwargs = mock_texas_holdem_class.call_args
        self.assertEqual(kwargs.get('num_players'), self.game_engine_config['num_players'])
        self.assertEqual(kwargs.get('starting_stack'), self.game_engine_config['starting_stack'])
        # ... check other params if necessary

        # Check that player_strategies were initialized (SelfPlay creates DummyStrategy)
        # This depends on how SelfPlay initializes TexasHoldem (it passes player_strategies)
        self.assertIn('player_strategies', kwargs)
        self.assertEqual(len(kwargs['player_strategies']), self.game_engine_config['num_players'])
        # Could add more detailed checks for DummyStrategy if needed

    @patch('self_play.self_play.TexasHoldem')
    @patch('self_play.self_play.torch.rand') # If state_representation uses torch.rand
    @patch('self_play.self_play.get_state_tensor_for_player') # Mock utility
    @patch('self_play.self_play.get_counterfactual_payoffs_for_player') # Mock utility
    def test_play_hand_for_training_basic_flow(self,
                                             mock_get_counterfactual_payoffs,
                                             mock_get_state_tensor,
                                             mock_torch_rand, # Unused if get_state_tensor is mocked well
                                             mock_texas_holdem_class):
        """
        Test the basic flow of play_hand_for_training, focusing on interactions.
        Simulates a very short hand: one player (AI) makes one decision.
        """
        # Setup mocks
        mock_game_engine = MagicMock()
        mock_texas_holdem_class.return_value = mock_game_engine

        # AI player is player 0
        ai_player_id = 0

        # Control game flow:
        # 1. Hand starts, not over. Current player is AI.
        # 2. AI plays, hand becomes over.
        mock_game_engine.is_hand_over.side_effect = [False, False, True] # Initial, after AI action, then terminate loop
        mock_game_engine.get_current_player_id.return_value = ai_player_id

        # Mock state and payoff data
        dummy_state_tensor = torch.randn(1, 10, 3) # batch, seq, feature
        dummy_counterfactual_payoffs = torch.randn(2) # num_actions

        mock_get_state_tensor.return_value = dummy_state_tensor
        mock_get_counterfactual_payoffs.return_value = dummy_counterfactual_payoffs

        # Mock AI's strategy and action choice
        # AICFRTrainer.model(state_tensor) -> strategy
        # Then action is chosen from strategy.
        # For simplicity, let's assume cfr_trainer.model.forward or similar is called by SelfPlay
        # to get strategy. Or SelfPlay calls something like cfr_trainer.get_action(state_tensor)
        # The current SelfPlay's play_action_for_player calls cfr_trainer.model to get strategy.
        mock_ai_strategy = torch.tensor([0.5, 0.5]) # Example 2 actions
        self.mock_cfr_trainer.model.return_value.squeeze.return_value = mock_ai_strategy

        # Action chosen by AI (e.g., first action)
        # self_play.py uses torch.multinomial to choose action.
        # We can patch torch.multinomial if we need to control the chosen action.
        with patch('torch.multinomial', return_value=torch.tensor([0])) as mock_multinomial:
            sp = SelfPlay(cfr_trainer=self.mock_cfr_trainer, game_engine_config=self.game_engine_config)
            sp.game_engine = mock_game_engine # Override with our detailed mock

            # Call the method
            training_data_collected = sp.play_hand_for_training()

        # Assertions
        mock_game_engine.reset_hand.assert_called_once()

        # AI player (0) should have played.
        # Check get_state_tensor was called for AI player
        mock_get_state_tensor.assert_called_with(mock_game_engine, ai_player_id)

        # Check model was called (via cfr_trainer)
        self.mock_cfr_trainer.model.assert_called_with(dummy_state_tensor.unsqueeze(0)) # Model expects batch

        # Check action was applied
        mock_multinomial.assert_called_once() # Action was chosen
        mock_game_engine.apply_action.assert_called_with(ai_player_id, 0) # Action 0 was chosen

        # Check that get_counterfactual_payoffs was called for the AI's state-action
        # This happens *after* the hand is over.
        # The current SelfPlay collects (state, chosen_action_probs, eventual_payoffs)
        # then calculates counterfactual_payoffs for each (state,action_i) pair *after* hand.
        # The `train` method is called with these.

        # The `cfr_trainer.train` is called with the state_tensor for the AI player
        # and the counterfactual payoffs calculated for that state.
        # This part is tricky to assert precisely without knowing the exact state representation
        # and payoff calculation logic within SelfPlay or its utilities.
        # We primarily check if `train` was called.
        self.mock_cfr_trainer.train.assert_called_once()

        # Check arguments of cfr_trainer.train()
        args, _ = self.mock_cfr_trainer.train.call_args
        called_state_tensor, called_payoffs = args

        self.assertTrue(torch.equal(called_state_tensor, dummy_state_tensor))
        # The `called_payoffs` in `train` are `all_counterfactual_payoffs`
        # which should be what `get_counterfactual_payoffs_for_player` returned.
        self.assertTrue(torch.equal(called_payoffs, dummy_counterfactual_payoffs))

        # Check that training data was collected
        self.assertTrue(len(training_data_collected) > 0) # At least one entry for the AI's decision
        # Example entry check:
        # data_entry = training_data_collected[0]
        # self.assertTrue(torch.equal(data_entry['state_tensor'], dummy_state_tensor))
        # self.assertTrue(torch.equal(data_entry['action_probabilities'], mock_ai_strategy))
        # self.assertIn('hand_payoff', data_entry) # Check if payoff was recorded

if __name__ == '__main__':
    unittest.main()
