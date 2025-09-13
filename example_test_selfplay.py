#!/usr/bin/env python3
"""Simple test script for self-play functionality"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing self-play imports...")

try:
    from poker_ai.selfplay.self_play import SelfPlay

    print("✓ SelfPlay imported successfully")

    # Test the mock components from the __main__ section
    import torch

    class MockModel(torch.nn.Module):
        def __init__(self, num_actions=10):
            super().__init__()
            self.num_actions = num_actions

        def forward(self, x):
            probs = torch.ones(1, self.num_actions) / self.num_actions
            return probs

    class MockCFRTrainer:
        def __init__(self, num_actions=10):
            self.config = {"model": {"max_seq_len": 20, "d_raw_feature": 3}}
            self.model = MockModel(num_actions=num_actions)
            self.num_actions = num_actions

        def train(self, state_tensor: torch.Tensor, all_counterfactual_payoffs: torch.Tensor):
            print(
                f"MockCFRTrainer.train called - State: {state_tensor.shape}, CF: {len(all_counterfactual_payoffs)}"
            )

    print("✓ Mock components created")

    # Test SelfPlay initialization
    mock_cfr_trainer = MockCFRTrainer(num_actions=10)
    game_config = {
        "num_players": 2,
        "starting_stack": 1000,
        "big_blind": 10,
        "small_blind": 5,
    }

    self_play_env = SelfPlay(cfr_trainer=mock_cfr_trainer, game_engine_config=game_config)
    print("✓ SelfPlay environment created")

    print("\nSelf-play test completed successfully! ✓")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
