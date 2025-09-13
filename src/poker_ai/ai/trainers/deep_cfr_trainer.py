import random

import torch
import torch.optim as optim

# Correctly import the refactored AdvantageNetwork
from poker_ai.ai.models.transformer import AdvantageNetwork


class ReplayBuffer:
    """A simple reservoir sampling replay buffer for Deep CFR."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list = []
        self.n_seen = 0

    def push(self, state: torch.Tensor, regrets: torch.Tensor, iteration: int):
        """Adds an experience to the buffer using reservoir sampling."""
        experience = (state.detach().cpu(), regrets.detach().cpu(), iteration)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            j = random.randint(0, self.n_seen)
            if j < self.capacity:
                self.buffer[j] = experience
        self.n_seen += 1

    def sample(self, batch_size: int) -> list:
        """Samples a batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DeepCFRTrainer:
    """
    A Deep CFR trainer that implements the algorithm from 'texas.tex'.
    It uses a single advantage network and trains with a weighted MSE loss (Linear CFR).
    """

    def __init__(
        self,
        input_feature_dim: int,
        hidden_dim: int,
        num_actions: int,
        learning_rate: float = 1e-4,
        buffer_capacity: int = 1_000_000,
        device: str | None = None,
    ):

        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_actions = num_actions
        self.card_feature_dim = input_feature_dim  # placeholder dimension

        # Use the new AdvantageNetwork; this trainer treats card summaries as zeros
        self.advantage_net = AdvantageNetwork(
            history_feature_dim=input_feature_dim,
            card_feature_dim=self.card_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            num_actions=num_actions,
        )
        self.advantage_net.to(self.device)

        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=learning_rate)

        # The replay buffer stores (infoset_embedding, realized_regrets, iteration_number)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # A minimal config dict for compatibility with other components
        self.config = {
            "model": {
                "d_raw_feature": input_feature_dim,
                "hidden_dim": hidden_dim,
                "num_actions": num_actions,
                "learning_rate": learning_rate,
            }
        }

    @torch.no_grad()
    def get_advantages(
        self, hole: torch.Tensor, community: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        """Return advantages conditioned on hole cards, community cards and history."""

        hole, community, history = (
            t.unsqueeze(0).to(self.device) if t.ndim == 1 else t.to(self.device)
            for t in (hole, community, history)
        )
        return self.advantage_net(hole, community, history).squeeze(0).cpu()

    def train(self, batch_size: int = 256):
        """
        Performs one training step on a batch from the replay buffer.
        This implements the weighted loss function from Linear CFR.
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample from the replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, regrets, iterations = zip(*batch, strict=False)

        states = torch.stack(states).to(self.device)
        regrets = torch.stack(regrets).to(self.device)
        iterations = torch.tensor(iterations, dtype=torch.float32, device=self.device).view(-1, 1)

        batch_size = states.size(0)
        zeros = torch.zeros(batch_size, self.card_feature_dim, device=self.device)

        # Get network predictions using zero card summaries
        adv_pred = self.advantage_net(zeros, zeros, states)

        # Calculate the weighted MSE loss (Linear CFR)
        # The loss is weighted by the iteration number T
        loss_values = (adv_pred - regrets) ** 2
        weighted_loss = (loss_values * iterations).sum() / iterations.sum()

        # Optimizer step
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        return weighted_loss.item()

    def save_model(self, path: str):
        """Saves the advantage network's state dict."""
        torch.save(self.advantage_net.state_dict(), path)

    def load_model(self, path: str):
        """Loads the advantage network's state dict."""
        self.advantage_net.load_state_dict(torch.load(path, map_location=self.device))
        self.advantage_net.to(self.device)
        self.advantage_net.eval()
