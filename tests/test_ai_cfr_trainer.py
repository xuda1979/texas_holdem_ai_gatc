import torch

from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer, AICFRReplayBuffer


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        # Parameter so that optimizers expecting parameters still function.
        self.logits = torch.nn.Parameter(torch.zeros(num_actions))

    def forward(
        self,
        hole_summary: torch.Tensor,
        community_summary: torch.Tensor,
        history_seq: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = hole_summary.shape[0]
        return self.logits.expand(batch, -1)


def test_train_single_scales_regrets_by_opponent_reach() -> None:
    trainer = AICFRTrainer(device="cpu")
    trainer.model = _ConstantLogitModel(trainer.num_actions)
    trainer.model.to(trainer.device)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)

    hole = torch.zeros(trainer.card_feature_dim)
    community = torch.zeros(trainer.card_feature_dim)
    history = torch.zeros(trainer.max_seq_len, trainer.history_feature_dim)
    payoffs = torch.linspace(-1.0, 1.0, steps=trainer.num_actions)

    opponent_reach = 0.25
    info_set_id = "test_infoset"

    trainer._train_single(
        info_set_id,
        hole,
        community,
        history,
        payoffs,
        opponent_reach=opponent_reach,
    )

    stored_regrets = trainer.cumulative_regret[info_set_id].detach().cpu()
    mean_payoff = payoffs.mean()
    expected = (payoffs - mean_payoff) * opponent_reach
    assert torch.allclose(stored_regrets, expected, atol=1e-6)


def test_replay_buffer_records_opponent_reach() -> None:
    buffer = AICFRReplayBuffer(capacity=2)
    hole = torch.zeros(2)
    community = torch.zeros(2)
    history = torch.zeros(1, 2)
    payoffs = torch.tensor([0.1, 0.2, 0.3])
    legal_mask = torch.tensor([True, False, True])

    buffer.push(
        hole,
        community,
        history,
        payoffs,
        legal_mask=legal_mask,
        opponent_reach=0.6,
        iteration=7,
    )

    entry = buffer.sample(1)[0]
    assert len(entry) == 7
    # Opponent reach should be preserved as a plain float for lightweight storage.
    assert entry[5] == 0.6
    assert entry[6] == 7
