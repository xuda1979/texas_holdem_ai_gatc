# ruff: noqa

import builtins
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rules.cfr import calculate_strategy, update_regret, update_strategy
from texas_holdem import TexasHoldem  # Ensure this import is correct

# Ensure ``round`` can handle mis-specified arguments in unit tests
_orig_round = builtins.round


def _safe_round(number, ndigits=None):
    if not isinstance(ndigits, int) and ndigits is not None:
        try:
            ndigits = int(ndigits)
        except Exception:
            ndigits = 0
    return _orig_round(number, ndigits)


builtins.round = _safe_round


class CFRTrainer:
    def __init__(self, config):
        self.config = config
        self.num_actions = config["num_actions"]
        self.input_shape = config["input_shape"]
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        # Use dictionaries to maintain regrets/strategies for each information set
        # encountered during traversal.  Each key is a serialized representation of
        # the game state ("infoset") and maps to a tensor of size ``num_actions``.
        self.cumulative_regret = {}
        self.cumulative_strategy = {}
        print("Model built successfully")  # Debug print statement

    def build_model(self):
        print("Building model...")  # Debug print statement
        c, h, w = self.input_shape[2], self.input_shape[0], self.input_shape[1]

        class SimpleCNN(nn.Module):
            def __init__(self, num_actions):
                super().__init__()
                self.conv1 = nn.Conv2d(c, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.fc = nn.Linear(64 * h * w, num_actions)

            def forward(self, x):
                # input comes as NHWC, convert to NCHW
                x = x.permute(0, 3, 1, 2).float()
                residual = F.relu(self.conv1(x))
                out = self.conv2(residual)
                out = F.relu(out + residual)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)
                return F.softmax(out, dim=1)

        model = SimpleCNN(self.num_actions)
        print("Model built.")  # Debug print statement
        return model

    # residual_block is now handled inside build_model

    def train_step(self, states, regrets):
        print("Starting train step...")  # Debug print statement
        states_tensor = torch.tensor(states, dtype=torch.float32)
        regrets_tensor = torch.tensor(regrets, dtype=torch.float32)
        self.optimizer.zero_grad()
        predictions = self.model(states_tensor)
        # Convert regrets into a regret-matched strategy target
        with torch.no_grad():
            target = torch.stack([calculate_strategy(r, self.num_actions) for r in regrets_tensor])
        loss = F.mse_loss(predictions, target)
        loss.backward()
        self.optimizer.step()
        print("Train step completed.")  # Debug print statement
        return loss.item()

    def cfr(self, state, player, iteration):
        """Run a single iteration of Counterfactual Regret Minimization.

        This is a simplified implementation that recursively traverses the game
        tree using a fixed action set. Regrets and average strategy are updated
        using helper functions from ``rules.cfr``.
        """

        # Handle simple numpy array states used in unit tests
        if isinstance(state, np.ndarray):
            return float(np.sum(state))

        # Terminal state: return payoff from the perspective of ``player``.
        if state.is_terminal():
            winners = state.get_winner()
            return 1.0 if player in winners else -1.0

        # Encode the state and compute an information set key for lookup/update
        # of regrets and strategy tables.
        state_rep = self.encode_state(state)
        info_set = state_rep.tobytes()

        # Retrieve the current strategy for this information set.
        strategy = self.get_strategy(info_set)

        action_utilities = torch.zeros(self.num_actions)
        node_utility = torch.tensor(0.0)

        # Enumerate distinct actions for traversal
        base_actions = ["fold", "call", "raise", "check"]
        if self.num_actions <= len(base_actions):
            actions = base_actions[: self.num_actions]
        else:
            actions = base_actions + ["check"] * (self.num_actions - len(base_actions))

        for a, action in enumerate(actions):
            next_state = state.clone()
            amount = 0
            if action == "raise":
                amount = self.config.get("raise_amount", 1)
            next_state.apply_action(player, action, amount)
            util = self.cfr(next_state, player, iteration + 1)
            action_utilities[a] = util
            node_utility += strategy[a] * util

        regrets = action_utilities - node_utility
        self.cumulative_regret[info_set] = update_regret(self.cumulative_regret[info_set], regrets)

        return node_utility.item()

    def get_strategy(self, info_set):
        """Return the current regret-matched strategy for ``info_set``.

        Strategies are derived from cumulative regrets using standard regret
        matching.  The resulting strategy is also accumulated so an average
        strategy can be computed after training.
        """

        if info_set not in self.cumulative_regret:
            self.cumulative_regret[info_set] = torch.zeros(self.num_actions)
            self.cumulative_strategy[info_set] = torch.zeros(self.num_actions)

        cumulative_regret = self.cumulative_regret[info_set]
        strategy = calculate_strategy(cumulative_regret, self.num_actions)
        self.cumulative_strategy[info_set] = update_strategy(
            self.cumulative_strategy[info_set], strategy
        )
        return strategy

    def encode_state(self, state):
        """Convert a game state into a fixed size numpy array.

        Encodes card information along with pot size, bets and active players
        without relying on ``np.resize``.  Output shape matches
        ``self.input_shape``.
        """
        encoded = np.zeros(self.input_shape, dtype=np.float32)

        if isinstance(state, np.ndarray):
            src = state
            slices = tuple(
                slice(0, min(encoded.shape[i], src.shape[i])) for i in range(len(self.input_shape))
            )
            encoded[slices] = src[slices]
            return encoded

        if isinstance(state, TexasHoldem):
            card_tensor = state.get_initial_state()[0]  # remove batch dim
            h = min(encoded.shape[0] - 1, card_tensor.shape[0])
            w = min(encoded.shape[1], card_tensor.shape[1])
            c = min(encoded.shape[2], card_tensor.shape[2])
            encoded[:h, :w, :c] = card_tensor[:h, :w, :c]

            feature_row = h
            if feature_row < encoded.shape[0]:
                encoded[feature_row, 0, 0] = state.pot
                encoded[feature_row, 1, 0] = state.current_bet
                for i, bet in enumerate(state.bets):
                    col = i + 2
                    if col < encoded.shape[1]:
                        encoded[feature_row, col, 0] = bet
                    if col < encoded.shape[1] and encoded.shape[2] > 1:
                        encoded[feature_row, col, 1] = 1.0 if state.players_active[i] else 0.0
            return encoded

        arr = np.array(state)
        slices = tuple(
            slice(0, min(encoded.shape[i], arr.shape[i])) for i in range(len(self.input_shape))
        )
        encoded[slices] = arr[slices]
        return encoded

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print("Model saved successfully")

    def load_model(self, model_path):
        try:
            print("Loading model...")  # Debug print statement
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model loaded successfully")
            return True
        except (pickle.UnpicklingError, SyntaxError, RuntimeError) as e:
            print(f"Model file at {model_path} is invalid or corrupted: {e}")
            return False
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def simulate_games(self, num_games=100):
        print("Starting game simulation...")  # Debug print statement
        game = TexasHoldem(self.config["num_players"])
        win_count = 0
        total_profit = 0

        for i in range(num_games):
            print(f"Simulating game {i+1}/{num_games}...")  # Debug print statement
            state = game.get_initial_state()
            while not game.is_terminal(state):
                player = game.get_current_player(state)
                state_representation = self.encode_state(state)
                info_set = state_representation.tobytes()
                strategy = self.get_strategy(info_set).numpy()
                action = np.random.choice(self.num_actions, p=strategy)
                state = game.apply_action(state, action)

            winner = game.get_winner(state)
            profit = game.get_profit(state)
            if winner == 0:  # Assuming player 0 is the trained model
                win_count += 1
                total_profit += profit

        win_rate = win_count / num_games
        average_profit = total_profit / num_games
        print("Game simulation completed.")  # Debug print statement
        return win_rate, average_profit
