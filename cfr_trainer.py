import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from texas_holdem import TexasHoldem

class CFRTrainer:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
        self.optimizer = Adam(learning_rate=config['learning_rate'])
        self.regret_sum = {}  # Tracks regrets for each action
        self.strategy_sum = {}  # Tracks cumulative strategies for average strategy computation

    def build_model(self):
        input_shape = (7, 13, 4)  # Match this to the shape of the state data
        input_layer = Input(shape=input_shape)

        x = Conv2D(64, kernel_size=3, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        for _ in range(self.config['num_res_blocks']):
            x = self.residual_block(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        output_layer = Dense(self.config['num_actions'], activation="linear")(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        return model


    def residual_block(self, x):
        shortcut = x
        x = Conv2D(64, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x

    def train_step(self, states, regrets):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = tf.reduce_mean(tf.square(predictions - regrets))  # Mean Squared Error as loss function
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss

    def cfr(self, game_state, player, iteration):
        if game_state.is_terminal():
            return game_state.get_payoff(player)

        state_representation = self.encode_state(game_state, player)
        strategy = self.get_strategy(state_representation)

        util = {action: 0 for action in game_state.get_actions()}
        node_utility = 0

        for action in game_state.get_actions():
            next_state = game_state.next_state(action)
            util[action] = -self.cfr(next_state, 1 - player, iteration)
            node_utility += strategy[action] * util[action]

        for action in game_state.get_actions():
            regret = util[action] - node_utility
            self.regret_sum[action] = self.regret_sum.get(action, 0) + regret

        return node_utility

    def get_strategy(self, state_representation):
        regret_sum = [self.regret_sum.get(action, 0) for action in range(self.config['num_actions'])]
        strategy = [max(r, 0) for r in regret_sum]
        normalizing_sum = sum(strategy)
        if normalizing_sum > 0:
            strategy = [s / normalizing_sum for s in strategy]
        else:
            strategy = [1.0 / len(strategy)] * len(strategy)
        return strategy

    def encode_state(self, game_state, player):
        # Encode the game state into a neural network input format
        # This is a placeholder; actual implementation will depend on the game state representation
        return game_state.get_encoded_state(player)

    def train(self, iterations):
        for iteration in range(iterations):
            game = TexasHoldem(self.config['num_players'])
            loss = self.train_step(game.get_initial_state(), self.regret_sum)
            print(f"Iteration {iteration}, Loss: {loss.numpy()}")

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
