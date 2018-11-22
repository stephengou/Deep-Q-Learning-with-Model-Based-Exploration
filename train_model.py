import gym
import time
from gym import wrappers
from DQN_Agent import DQN_Agent
from Random_Agent import Random_Agent
from DQN_Dynamics import DQN_Dynamics
from DQN_Least_Likely_Action import  DQN_Least_Likely_Action
from DQN_Dynamics_Normal import DQN_Dynamics_Normal
from DQN_Mem_Gen import  DQN_Mem_Gen
from Helpers import plot_rewards_and_length,plot_state_scatter
import numpy as np

env_name = "MountainCar-v0"#"Acrobot-v1"#"LunarLander-v2"#"BipedalWalker-v2"#"CartPole-v0"#"HalfCheetah-v2"#MountainCar-v0
max_episodes = 75
record_video_every = 50

def main():
    env = gym.make(env_name)
    env = wrappers.Monitor(env, 'replay', video_callable=lambda e: e%record_video_every == 0,force=True)

    state_shape = (1,env.observation_space.shape[0])
    agent = DQN_Dynamics_Normal(env=env)
    start_time = time.time()
    total_reward_list = []
    episode_length_list = []
    for episode in range(max_episodes):
        agent.on_episode_start()
        cur_state = env.reset().reshape(state_shape)
        steps = 0
        total_reward = 0
        done = False
        while not done:
            steps += 1
            action = agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(state_shape)
            agent.update_model(cur_state, action, reward, new_state, done)
            cur_state = new_state
            total_reward += reward
            if done:
                break

        agent.on_episode_end()
        total_reward_list.append(total_reward)
        episode_length_list.append(steps)
        print('episode {} steps: {}, total reward: {},  elapsed time: {}s'.format(episode, steps, total_reward, int(time.time()-start_time)))

    plot_state_scatter(agent)
    plot_rewards_and_length(total_reward_list, episode_length_list)

if __name__ == "__main__":
    main()