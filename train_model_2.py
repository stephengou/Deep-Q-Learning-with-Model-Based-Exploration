import gym
import time
from gym import wrappers
from Helpers import plot_rewards_and_length,plot_state_scatter
from PolicyGradient import PG_MonteCarlo


env_name = "CartPole-v0"
max_episodes = 75
record_video_every = 50


def main():
    env = gym.make(env_name)
    env = wrappers.Monitor(env, 'replay', video_callable=lambda e: e%record_video_every == 0,force=True)

    state_shape = (1,env.observation_space.shape[0])
    agent = PG_MonteCarlo.PG_MonteCarlo(env=env)

    agent.generate_model()
    agent.train()

if __name__ == "__main__":
    main()