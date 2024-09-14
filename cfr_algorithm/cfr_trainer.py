import tensorflow as tf

class CFRTrainer:
    def __init__(self, model, num_actions, learning_rate=0.001):
        self.model = model
        self.num_actions = num_actions
        self.regrets = tf.Variable(tf.zeros([num_actions]), trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def get_strategy(self):
        positive_regrets = tf.maximum(self.regrets, 0)
        if tf.reduce_sum(positive_regrets) > 0:
            return positive_regrets / tf.reduce_sum(positive_regrets)
        else:
            return tf.ones([self.num_actions]) / self.num_actions

    def update_regret(self, action, regret_value):
        updated_regrets = tf.tensor_scatter_nd_update(self.regrets, [[action]], [self.regrets[action] + regret_value])
        self.regrets.assign(updated_regrets)

    def update_strategy(self, states, regrets):
        # Forward pass through the model
        with tf.GradientTape() as tape:
            strategy_pred = self.model(states)
            loss = tf.reduce_mean(tf.keras.losses.MSE(strategy_pred, regrets))

        # Backpropagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

# Example usage
if __name__ == "__main__":
    # Dummy model for demonstration purposes
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    trainer = CFRTrainer(model, num_actions=4)
    trainer.update_regret(2, 1.0)
    print(trainer.get_strategy())