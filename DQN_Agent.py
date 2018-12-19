#reference: https://keon.io/deep-q-learning/
# this dqn is used as a baseline and starter code for our improved DQN
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from Agent import Agent

class DQN_Agent(Agent):
    def __init__(self, env):
        self.env = env
        self.replay_memory = deque(maxlen=200000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.05
        self.target_update_counter = 0
        self.C = 8 # intervcal for updating target network
        self.initial_random_steps = 0
        self.actions_count = 0
        self.clip_errors = True

        self.q_network = self.init_q_network()
        self.target_q_network = self.init_q_network()

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def get_observation_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space

    def init_q_network(self):
        model = Sequential()
        state_shape = self.get_observation_space().shape
        model.add(Dense(48, input_shape=state_shape, activation="relu"))
        #model.add(Dense(48, activation="relu"))
        #model.add(Dense(24, activation="relu"))
        model.add(Dense(self.get_action_space().n, activation='linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.actions_count += 1
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon or self.actions_count < self.initial_random_steps:
            return self.get_action_space().sample()
        return np.argmax(self.q_network.predict(state)[0])

    def update_model(self, state, action, reward, new_state, done):
        self.replay_memory.append([state, action, reward, new_state, done])
        self.fit_q_network()
        self.update_target_q_network()

    def sample_replays(self,batch_size):
        return random.sample(self.replay_memory, batch_size)

    def fit_q_network(self):
        #sample replay and do SGD
        batch_size = 16
        if len(self.replay_memory) < batch_size:
            return

        samples = self.sample_replays(batch_size)
        sampled_states = []
        sampled_targets = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_q_network.predict(state)
            predicted = self.q_network.predict(state)
            if done:
                target[0][action] = reward
            else:
                #update target by Bellman equation
                target[0][action] = reward + self.gamma * max(self.target_q_network.predict(new_state)[0])

                if self.clip_errors:
                    #clip error to -1, +1
                    if (target[0][action] > predicted[0][action]):
                        target[0][action] = predicted[0][action] + 1
                    elif (target[0][action] > predicted[0][action]):
                        target[0][action] = predicted[0][action] - 1
            sampled_states.append(state)
            sampled_targets.append(target)

        batched_states = np.concatenate(sampled_states,axis=0)
        batched_targets = np.concatenate(sampled_targets,axis=0)
        self.q_network.fit(batched_states, batched_targets, epochs=1, verbose=0)

    def update_target_q_network(self):
        #update target q network every C steps
        self.target_update_counter += 1
        if (self.target_update_counter > self.C):
            self.target_update_counter = 0
            self.target_q_network.set_weights(self.q_network.get_weights())
