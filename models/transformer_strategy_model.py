# transformerAIStrategy.py

import tensorflow as tf
from tensorflow.keras import layers
from playStrategy import PlayerStrategy
import numpy as np

class TransformerStrategyModel(tf.keras.Model):
    def __init__(
        self,
        num_actions,
        d_model=128,
        num_heads=8,
        num_layers=3,
        dim_feedforward=512,
        max_seq_length=1
    ):
        super(TransformerStrategyModel, self).__init__()
        self.num_actions = num_actions
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length  # Single game state

        # Input projection
        self.input_dense = layers.Dense(d_model, activation='relu')

        # Positional Encoding (optional for single sequence input)
        self.positional_encoding = self.add_weight(
            shape=(1, max_seq_length, d_model),
            initializer='zeros',
            trainable=True
        )

        # Transformer Encoder Layers
        self.transformer_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            for _ in range(num_layers)
        ]
        self.transformer_norms = [
            layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)
        ]

        # Feed-Forward Network
        self.feed_forward = layers.Dense(dim_feedforward, activation='relu')
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)

        # Output Layers
        self.action_output = layers.Dense(num_actions, activation='softmax', name='action_output')
        self.amount_output = layers.Dense(1, activation='linear', name='amount_output')

    def call(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, feature_dim)
        """
        x = self.input_dense(x)  # (batch_size, seq_length, d_model)
        x += self.positional_encoding[:, :tf.shape(x)[1], :]  # Add positional encoding

        for i in range(self.num_layers):
            # Multi-Head Attention
            attn_output = self.transformer_layers[i](x, x)
            attn_output = self.transformer_norms[i](x + attn_output)  # Residual connection and normalization

            # Feed-Forward Network
            ffn_output = self.feed_forward(attn_output)
            ffn_output = self.ffn_norm(attn_output + ffn_output)  # Residual connection and normalization

            x = ffn_output  # Update x for next layer

        # Separate outputs
        action_probs = self.action_output(x)  # (batch_size, seq_length, num_actions)
        amount_pred = self.amount_output(x)    # (batch_size, seq_length, 1)

        return action_probs, amount_pred

    def get_config(self):
        config = super(TransformerStrategyModel, self).get_config()
        config.update({
            'num_actions': self.num_actions,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'max_seq_length': self.max_seq_length
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerAIStrategy(PlayerStrategy):
    def __init__(
        self,
        num_actions=6,  # Example: 0-Fold, 1-Check, 2-Call, 3-Bet, 4-Raise, 5-All-In
        d_model=128,
        num_heads=8,
        num_layers=3,
        dim_feedforward=512,
        min_bet=10,
        max_bet=100
    ):
        super().__init__()
        self.num_actions = num_actions
        self.min_bet = min_bet
        self.max_bet = max_bet

        self.model = TransformerStrategyModel(
            num_actions=num_actions,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_seq_length=1  # Single game state
        )

        # Optionally, load pretrained weights
        # Example:
        # self.model.load_weights('path_to_pretrained_weights.h5')

    @property
    def is_human(self):
        return False

    def choose_action(self, game, player_index):
        # Extract game state features
        features = self.extract_features(game, player_index)
        features = np.expand_dims(features, axis=0)  # Shape: (1, feature_dim)
        features = np.expand_dims(features, axis=1)  # Shape: (1, 1, feature_dim)

        # Convert to Tensor
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

        # Get action probabilities and amount prediction from the model
        action_probs, amount_pred = self.model(features_tensor)
        action_probs = action_probs.numpy()[0][0]  # Shape: (num_actions,)
        amount_pred = amount_pred.numpy()[0][0]    # Scalar

        # Select action based on probabilities
        action = np.random.choice(self.num_actions, p=action_probs)
        action_map = {
            0: 'fold',
            1: 'check',
            2: 'call',
            3: 'bet',
            4: 'raise',
            5: 'all-in'
        }
        selected_action = action_map.get(action, 'fold')  # Default to 'fold' if mapping fails

        # If action is 'bet' or 'raise', decide raise amount
        if selected_action in ['bet', 'raise']:
            # Normalize amount_pred to be within [min_bet, max_bet]
            amount = int(np.clip(amount_pred, self.min_bet, self.max_bet))
            # Ensure amount does not exceed player's available chips
            available_chips = game.rules.player_chips[player_index]
            current_bet = game.rules.current_bet
            player_current_bet = game.rules.bets[player_index]
            required_call = current_bet - player_current_bet
            max_possible_raise = available_chips - required_call
            if selected_action == 'raise':
                # In a raise, the amount should be at least the min_raise
                min_raise = game.get_min_raise_amount(player_index)
                if amount < min_raise:
                    amount = min_raise
                amount = int(np.clip(amount, min_raise, max_possible_raise))
            elif selected_action == 'bet':
                # In a bet, the amount is the initial bet
                amount = int(np.clip(amount, self.min_bet, max_possible_raise))
            return selected_action, amount
        else:
            return selected_action, None

    def extract_features(self, game, player_index):
        """
        Extract and encode the game state into a feature vector.
        """
        # Player's current chips
        player_chips = game.rules.player_chips[player_index]

        # Current bet in the game
        current_bet = game.rules.current_bet

        # Player's current bet
        player_bet = game.rules.bets[player_index]

        # Encode community cards
        community_cards = game.rules.community_cards
        num_community_cards = len(community_cards)
        max_community_cards = 5
        card_encoding = []
        for card in community_cards:
            rank = self.card_rank_to_int(card[0])
            suit = self.card_suit_to_int(card[1])
            card_encoding.extend([rank, suit])
        while len(card_encoding) < max_community_cards * 2:
            card_encoding.extend([0, 0])  # Padding

        # Encode player's hole cards
        hole_cards = game.rules.hands[player_index]
        hole_card_encoding = []
        for card in hole_cards:
            rank = self.card_rank_to_int(card[0])
            suit = self.card_suit_to_int(card[1])
            hole_card_encoding.extend([rank, suit])
        while len(hole_card_encoding) < 2 * 2:
            hole_card_encoding.extend([0, 0])  # Padding for two cards

        # Combine all features into a single vector
        features = [
            player_chips,
            current_bet,
            player_bet
        ] + card_encoding + hole_card_encoding

        # Normalize features if necessary (optional)
        # Example: scale chips and bets by dividing by a constant
        # Here, assuming starting_stack = 10000
        features = [
            player_chips / 10000,
            current_bet / 10000,
            player_bet / 10000
        ] + card_encoding + hole_card_encoding

        return np.array(features, dtype=np.float32)

    def card_rank_to_int(self, rank):
        rank_dict = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_dict.get(rank.upper(), 0)

    def card_suit_to_int(self, suit):
        suit_dict = {'h': 1, 'd': 2, 'c': 3, 's': 4}
        return suit_dict.get(suit.lower(), 0)
