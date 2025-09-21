import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.ai.models.transformer import AdvantageNetwork
from poker_ai.cli.play_vs_ai import AIStrategy


class TestInferenceMultiNPU(unittest.TestCase):
    def test_ai_strategy_dataparallel(self):
        mock_npu = MagicMock()
        mock_npu.is_available.return_value = True
        mock_npu.device_count.return_value = 4

        dummy_state = AdvantageNetwork(
            history_feature_dim=18,
            card_feature_dim=18,
            hidden_dim=128,
            num_heads=4,
            num_layers=AdvantageNetwork.DEFAULT_NUM_LAYERS,
            num_actions=10,
        ).state_dict()

        with (
            patch.object(torch, "npu", mock_npu, create=True),
            patch("torch.nn.Module.to", lambda self, *args, **kwargs: self),
            patch("torch.load", return_value=dummy_state),
        ):
            strategy = AIStrategy("dummy.pth", "npu", use_all_npus=True)
            self.assertIsInstance(strategy.model, torch.nn.DataParallel)


if __name__ == "__main__":
    unittest.main()
