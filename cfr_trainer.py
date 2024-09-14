import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation, add
from tensorflow.keras.models import Model
import numpy as np
from texas_holdem import TexasHoldem  # Ensure this import is correct

class CFRTrainer:
    def __init__(self, config):
        self.config = config
        self.num_actions = config['num_actions']
        self.input_shape = config['input_shape']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        self.model = self.build_model()
        self.regrets = {}
        self.strategy = {}
        print("Model built successfully")  # Debug print statement

    def build_model(self):
        print("Building model...")  # Debug print statement
        inputs = Input(shape=self.input_shape)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = self.residual_block(x)
        x = Flatten()(x)
        outputs = Dense(self.num_actions, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss='mse')
        print("Model built.")  # Debug print statement
        return model

    def residual_block(self, x):
        print("Adding residual block...")  # Debug print statement
        shortcut = x
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = add([x, shortcut])
        x = Activation('relu')(x)
        print("Residual block added.")  # Debug print statement
        return x

    def train_step(self, states, regrets):
        print("Starting train step...")  # Debug print statement
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = tf.keras.losses.mean_squared_error(regrets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        print("Train step completed.")  # Debug print statement
        return loss.numpy().mean()

    def cfr(self, state, player, iteration):
        # Implement the CFR algorithm here
        pass

    def get_strategy(self, state_representation):
        print("Getting strategy...")  # Debug print statement
        predictions = self.model.predict(np.array([state_representation]))[0]
        strategy = predictions / np.sum(predictions)  # Normalize to get probabilities
        print("Strategy obtained.")  # Debug print statement
        return strategy

    def encode_state(self, state):
        # Implement state encoding here
        pass

    def save_model(self, model_path):
        self.model.save(model_path)
        print("Model saved successfully")

    def load_model(self, model_path):
        try:
            print("Loading model...")  # Debug print statement
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def simulate_games(self, num_games=100):
        print("Starting game simulation...")  # Debug print statement
        game = TexasHoldem(self.config['num_players'])
        win_count = 0
        total_profit = 0

        for i in range(num_games):
            print(f"Simulating game {i+1}/{num_games}...")  # Debug print statement
            state = game.get_initial_state()
            while not game.is_terminal(state):
                player = game.get_current_player(state)
                state_representation = self.encode_state(state)
                strategy = self.get_strategy(state_representation)
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

# Example usage
if __name__ == "__main__":
    print("Starting script...")  # Debug print statement
    config = {
        'num_actions': 10,  # Example number of actions
        'learning_rate': 0.001,
        'input_shape': (8, 8, 3),  # Example input shape
        'num_players': 2  # Example number of players
    }
    print("Config created:", config)  # Debug print statement
    trainer = CFRTrainer(config)
    print("Loading model...")  # Debug print statement
    trainer.load_model('trained_model.h5')
    print("Simulating games...")  # Debug print statement
    win_rate, average_profit = trainer.simulate_games(num_games=100)
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Average Profit: {average_profit:.2f}")