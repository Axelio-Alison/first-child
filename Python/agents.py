import numpy as np
import pandas as pd
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Add, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
import json
import random
import keras


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
        self.epsilon_min = 0.025
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

    def act(self, state, training = True):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon and training:
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

    def load(self, filepath):
        self.model = load_model(filepath + "\\model.keras")
        self.target_model = load_model(filepath + "\\target_model.keras")

        self.memory = deque(pd.read_pickle(filepath + "\\memory.pkl").to_records(index = False))

        episode = -1

        with open(filepath + "\\agent_data.json", 'r') as f:
            data = json.load(f)

            self.state_size = data['state_size']
            self.action_size = data['action_size']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon']
            self.epsilon_min = data['epsilon_min']
            self.epsilon_decay = data['epsilon_decay']
            self.learning_rate = data['lr']

            episode = data['episode']
        
        return episode



    def save(self, filepath, episode):
        self.model.save(filepath + "\\model.keras")
        self.target_model.save(filepath + "\\target_model.keras")

        memory_df = pd.DataFrame(self.memory, columns = ['state', 'action', 'reward', 'next_state', 'done'])
        memory_df.to_pickle(filepath + "\\memory.pkl")

        data = {'state_size' : int(self.state_size), 'action_size' : int(self.action_size), 'episode' : episode, 'gamma' : float(self.gamma), 
        'epsilon' : float(self.epsilon), 'epsilon_min' : float(self.epsilon_min), 'epsilon_decay' : float(self.epsilon_decay), 'lr' : float(self.learning_rate)}

        with open(filepath + "\\agent_data.json", 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)




@keras.saving.register_keras_serializable(package = "MyDQNLayers")
class AdvantageNormalization(Layer):
    def __init__(self, **kwargs):
        super(AdvantageNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        advantage = inputs
        mean_advantage = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return advantage - mean_advantage

@keras.saving.register_keras_serializable(package="MyDQNLayers", name="_huber_loss")
def _huber_loss(y_true, y_pred):
    err = y_true - y_pred
    cond = tf.abs(err) < 1.0
    L2 = 0.5 * tf.square(err)
    L1 = (tf.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return tf.reduce_mean(loss)

class D3QNAgent(DDQNAgent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    

    def _build_model(self):
        input_layer = Input(shape=(self.state_size,))
        dense1 = Dense(75, activation='relu')(input_layer)
        dense2 = Dense(70, activation='relu')(dense1)
        value_fc = Dense(66, activation='relu')(dense2)
        advantage_fc = Dense(66, activation='relu')(dense2)
        value = Dense(1)(value_fc)
        advantage = Dense(self.action_size)(advantage_fc)
        advantage_normalized = AdvantageNormalization()(advantage)
        q_value = Add()([value, advantage_normalized])
        model = Model(inputs=input_layer, outputs=q_value)
        model.compile(loss=_huber_loss, optimizer=Adam(learning_rate=self.learning_rate))
        #model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model






