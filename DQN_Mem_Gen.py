import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from DQN_Agent import DQN_Agent

class DQN_Mem_Gen(DQN_Agent):
    def __init__(self, env):
        self.env = env
        self.replay_memory = deque(maxlen=200000)
        self.positive_memory = deque(maxlen=5000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.target_update_counter = 0
        self.C = 8 # intervcal for updating target network
        self.initial_random_steps = 500
        self.actions_count = 0
        self.clip_errors = True

        self.q_network = self.init_q_network()
        self.target_q_network = self.init_q_network()
        self.dynamics_model = self.init_dynamics_model()
        self.update_count = 0
        self.dynamics_model_converged = False

    def generate_memories(self):
        #inject positive memories
        artificial_mem = [np.array([0.486,0.013]).reshape(1,-1), 2, -1.,np.array([0.5, 0.015]).reshape(1,-1), True]
        self.positive_memory.append(artificial_mem)
        for a in range(3):
            state = artificial_mem[0]
            state_prev_a = np.append(state, [[float(a)]], axis=1)
            prev_state = self.dynamics_model.predict([state_prev_a])
            if prev_state[0][0] > 5. or prev_state[0][0] < -1.23 or prev_state[0][1] > 0.06 or prev_state[0][1] < -0.06:
                continue
            memory = [prev_state,a, self.predict_reward(prev_state),state, False]
            self.positive_memory.append(memory)
        print('generated memories!' + str(len(self.positive_memory)))
        print(self.positive_memory)

    def update_model(self, state, action, reward, new_state, done):

        self.replay_memory.append([state, action, reward, new_state, done])
        self.fit_q_network()
        self.update_target_q_network()
        self.update_count += 1

        if self.dynamics_model_converged:
            if len(self.positive_memory) < 2:
                self.generate_memories()
            return

        if self.update_count % 50 == 0:
            self.fit_dynamics_model()
        if self.update_count % 500 == 0:
            self.eval_dynamics_model()

    def act(self, state):
        self.actions_count += 1
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon or self.actions_count < self.initial_random_steps:
            return self.explore(state)
        return np.argmax(self.q_network.predict(state)[0])

    def explore(self,state):
        return self.get_action_space().sample()

    def predict_reward(self,state):
        #TODO
        return -1.

    def sample_replays(self,batch_size):
        positive_batch = int(batch_size * 0.3)
        if len(self.positive_memory) < positive_batch:
            positive_batch = len(self.positive_memory)

        non_positive_batch = batch_size - positive_batch
        return random.sample(self.replay_memory, non_positive_batch) + random.sample(self.positive_memory, positive_batch)

    def init_dynamics_model(self):
        model = Sequential()
        state_shape = (self.get_observation_space().shape[0] + 1,)
        print(state_shape)
        model.add(Dense(24, input_shape=state_shape, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.get_observation_space().shape[0], activation='linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.002))
        return model

    def fit_dynamics_model(self):
        batchsize = 64
        if len(self.replay_memory) < batchsize:
            return
        samples = self.sample_replays(batchsize)
        sampled_states = []
        sampled_next_states = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            input_state = state
            next_state = np.append(new_state, [[action]], axis=1)
            sampled_states.append(input_state)
            sampled_next_states.append(next_state)

        batched_states = np.concatenate(sampled_states, axis=0)
        batched_next_states = np.concatenate(sampled_next_states, axis=0)
        self.dynamics_model.fit(batched_next_states, batched_states, epochs=2, verbose=0)

    #debug use only
    def eval_dynamics_model(self):
        samples = self.sample_replays(32)
        sampled_states = []
        sampled_targets = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            input_state = state
            target = np.append(new_state, [[action]], axis=1)
            sampled_states.append(input_state)
            sampled_targets.append(target)

        batched_inputs = np.concatenate(sampled_states, axis=0)
        batched_targets = np.concatenate(sampled_targets, axis=0)
        scores = self.dynamics_model.evaluate(batched_targets,batched_inputs,verbose=0)
        if scores < 0.001:
            self.dynamics_model_converged = True
            print('Dynamics model has converged!')
            #for j in range(10):
                #print('next state: ' + str(sampled_targets[j]) + 'prev state: ' + str(self.dynamics_model.predict([sampled_targets[j]])))
        print(self.dynamics_model.metrics_names, scores)
