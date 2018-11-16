import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from DQN_Agent import DQN_Agent

class DQN_PrioritizedReplay(DQN_Agent):
    def __init__(self, env):
        self.env = env
        self.replay_memory = deque(maxlen=200000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.target_update_counter = 0
        self.C = 8 # intervcal for updating target network
        self.initial_random_steps = 0
        self.actions_count = 0
        self.clip_errors = True

        self.q_network = self.init_q_network()
        self.target_q_network = self.init_q_network()

    def sample_replays(self,batch_size):
        return random.sample(self.replay_memory, batch_size)