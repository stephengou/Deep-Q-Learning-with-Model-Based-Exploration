from DQN_Agent import  DQN_Agent
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from Agent import Agent

class Multiagent_DQN(DQN_Agent):
    def __init__(self,env,id,clip_errors = True):
        self.id = id
        DQN_Agent.__init__(self,env)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.02
        self.target_update_counter = 0
        self.C = 10
        self.clip_errors = clip_errors
        self.initial_random_steps = 1000

    def get_observation_space(self):
        return self.env.observation_space[self.id]

    def get_action_space(self):
        return self.env.action_space[self.id]

    def init_q_network(self):
        model = Sequential()
        state_shape = self.get_observation_space().shape
        model.add(Dense(24, input_shape=state_shape, activation="relu"))
        #model.add(Dense(24, input_shape=state_shape, activation="relu"))
        #model.add(Dense(24, input_shape=state_shape, activation="relu"))
        #model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.get_action_space().n, activation='linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model