#reference: https://keon.io/deep-q-learning/
# this dqn is used as a baseline and basic model for our independent multiagent DQN
from Agent import Agent

#benchmark: this initial parameters setting can solve cartpole in ~40 episodes

class Random_Agent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, state):
        return self.env.action_space.sample()