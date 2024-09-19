import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class TransformerStrategyModel(tf.keras.Model):
    def __init__(self, num_actions, d_model=128, num_heads=8, num_layers=3, dim_feedforward=512, max_seq_length=1):
        super(TransformerStrategyModel, self).__init__()
        self.num_actions = num_actions
        self.input_projection = layers.Dense(d_model, activation='relu')  # Input projection layer
        self.positional_encoding = self.add_weight(shape=(1, max_seq_length, d_model), initializer='zeros', trainable=True)
        self.transformer_layers = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model) for _ in range(num_layers)]
        self.transformer_norms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.feed_forward = layers.Dense(dim_feedforward, activation='relu')
        self.output_projection = layers.Dense(d_model)  # Add this layer to project back to d_model size
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)
        self.action_output = layers.Dense(num_actions, activation='softmax', name='action_output')
        self.amount_output = layers.Dense(1, activation='linear', name='amount_output')

    def call(self, x):
        # Project the smaller input into the expected d_model dimensionality
        x = self.input_projection(x)  # Project to d_model size
        x += self.positional_encoding[:, :tf.shape(x)[1], :]
        for i in range(len(self.transformer_layers)):
            x = self._transformer_layer(x, i)
        return self.action_output(x), self.amount_output(x)

    def _transformer_layer(self, x, i):
        attn_output = self.transformer_layers[i](x, x)
        attn_output = self.transformer_norms[i](x + attn_output)
        ffn_output = self.feed_forward(attn_output)
        ffn_output = self.output_projection(ffn_output)  # Project the feed-forward output back to d_model size
        return self.ffn_norm(attn_output + ffn_output)



    def get_config(self):
        return {
            'num_actions': self.num_actions,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# AI Strategy Class that Uses the Transformer Model
class TransformerAIStrategy:
    def __init__(self, num_actions=6):
        self.num_actions = num_actions
        self.model = TransformerStrategyModel(num_actions=num_actions)

    @property
    def is_human(self):
        return False  # AI-controlled player

    def choose_action(self, game_state, player_index):
        """
        Chooses an action based on the current game state.
        
        Parameters:
        - game_state: A dictionary representing the current state of the game.
        - player_index: Index of the player making the decision.

        Returns:
        - action: The chosen action (fold, check, call, raise, all-in).
        - amount: The amount to bet or raise, if applicable.
        """
        # Extract game features based on the current game state
        features = self._get_game_features(game_state, player_index)
        return self._process_action(features, game_state, player_index)

    def _card_to_numeric(self, card):
        """
        Converts a card string like 'Js' (Jack of Spades) into a numerical value.
        Suits are mapped to numbers and ranks are mapped to numbers.
        """
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'h': 1, 'd': 2, 'c': 3, 's': 4}

        # Extract rank and suit from the card string
        rank = card[0]
        suit = card[1]
        
        # Convert the rank and suit to numerical values and return as a float
        return rank_map[rank] + (suit_map[suit] / 10)  # Slightly adjust suit for uniqueness


    def _get_game_features(self, game_state, player_index):
        """
        Extracts features from the game state for the model.
        
        Parameters:
        - game_state: A TexasHoldem object with relevant game information.
        - player_index: The index of the player for whom features are being extracted.
        
        Returns:
        - A tensor representing the input features for the model.
        """
        # Accessing the player's hand and community cards from game_state.rules
        player_hand = np.array([self._card_to_numeric(card) for card in game_state.rules.hands[player_index]])  # Player hand
        community_cards = np.array([self._card_to_numeric(card) for card in game_state.rules.community_cards])  # Community cards
        
        # Concatenate player hand and community cards
        features = np.concatenate((player_hand, community_cards), axis=None)
        
        # Return features as a tensor
        return tf.convert_to_tensor(features, dtype=tf.float32)



    def _process_action(self, features, game_state, player_index):
        """
        Processes the model output and validates the action within the game constraints.
        
        Parameters:
        - features: The extracted features for the player and game state.
        - game_state: The current state of the game.
        - player_index: The index of the player making the decision.
        
        Returns:
        - action: A validated and game-compliant action.
        - amount: A validated amount for betting or raising.
        """
        # Pass the extracted features to the model and get the action and amount
        features = tf.expand_dims(features, axis=0)  # Add batch dimension
        action_probs, amount_pred = self.model(features)
        action = tf.argmax(action_probs[0, 0], axis=-1).numpy()
        amount = amount_pred[0, 0].numpy()
        
        # Validate and adjust the action to satisfy game rules
        return self._validate_action(game_state, player_index, action, amount)



    def _validate_action(self, game_state, player_index, action, amount):
        """
        Validates and adjusts the model's predicted action based on game rules.
        
        Parameters:
        - game_state: The current state of the game (TexasHoldem object).
        - player_index: The index of the player making the decision.
        - action: The action predicted by the model (fold, check, call, raise, all-in).
        - amount: The amount predicted by the model for betting or raising.
        
        Returns:
        - action: A validated action (fold, check, call, raise, all-in).
        - amount: A validated and adjusted amount.
        """
        # Calculate min and max bet based on game rules (adjust these as necessary)
        min_raise = game_state.rules.current_bet  # Replace this with how the minimum raise is calculated
        max_raise = game_state.rules.player_chips[player_index]  # Maximum raise is all the player's chips
        amount_to_call = game_state.rules.current_bet - game_state.rules.bets[player_index]
        player_chips = game_state.rules.player_chips[player_index]

        # Ensure the action makes sense given the game state
        if action == 0:  # Fold
            return 'fold', None
        elif action == 1:  # Check
            # Player can only check if there's no amount to call
            if amount_to_call > 0:
                action = 2  # Change to call
            return 'check' if amount_to_call == 0 else 'call', amount_to_call
        elif action == 2:  # Call
            if amount_to_call > player_chips:
                action = 0  # If they can't call, they must fold
                return 'fold', None
            return 'call', amount_to_call
        elif action in [3, 4]:  # Bet or Raise
            # Ensure bet/raise amount is within valid limits
            if amount < min_raise:
                amount = min_raise
            elif amount > max_raise:
                amount = max_raise
            if amount > player_chips:
                amount = player_chips  # Cannot raise more than the player's chips
            return 'raise', amount
        elif action == 5:  # All-In
            return 'all-in', player_chips
        else:
            raise ValueError(f"Unsupported action index: {action}")
