import numpy as np
from collections import namedtuple, deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random


import tensorflow as tf
from tensorflow.keras.layers import Layer

# Useless function
class SumZeroSoftmaxLayer(Layer):
    def __init__(self, **kwargs):
        super(SumZeroSoftmaxLayer, self).__init__(**kwargs)

    def call(self, inputs):
        softmax_outputs = tf.nn.softmax(inputs, axis=-1)
        mean_softmax_outputs = tf.reduce_mean(softmax_outputs, axis=-1, keepdims=True)
        return softmax_outputs - mean_softmax_outputs


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()  # Online network
        self.target_model = self._build_model()  # Target network
        self.update_target_model()

    def _build_model(self):
        # Neural network to approximate Q-function
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from the online network to the target network
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose = 0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Sample a batch of experiences from the replay buffer
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose = 0)

            if done:
                # If done, the reward is the final value
                target[0][action] = reward
            else:
                # Double DQN update
                # Online network selects the action
                a = np.argmax(self.model.predict(next_state, verbose = 0)[0])
                
                # Target network evaluates the action
                t = self.target_model.predict(next_state, verbose = 0)[0][a]
                
                # Update the target value
                target[0][action] = reward + self.gamma * t
                
            
            # Train the online network
            self.model.fit(state, target, epochs=1, verbose=0)
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name, verbose = 0)

    def save(self, name):
        self.model.save_weights(name)