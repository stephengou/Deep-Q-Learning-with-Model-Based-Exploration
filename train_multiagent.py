import time
from Multiagent_DQN import Multiagent_DQN
from Random_Agent import Random_Agent
import numpy as np
from Helpers import plot_rewards_and_length,make_multi_env

max_episodes = 500
max_steps = 30
render_every = 1

def done_callback(agent, world):
    total_d = 0
    for entity in world.landmarks:
        total_d += np.linalg.norm(entity.state.p_pos - agent.state.p_pos)

    print(total_d)
    return total_d < 3. or total_d > 200.

def main():
    env = make_multi_env('simple',False,done_cb = done_callback)
    env.discrete_action_input = True
    #num of agents
    N = env.n
    agent_list = [Multiagent_DQN(env=env,id=i,clip_errors=True) for i in range(N)]
    start_time = time.time()
    total_reward_list = []
    episode_length_list = []
    for episode in range(max_episodes):
        cur_state_list = [env.reset()[i].reshape((1, env.observation_space[i].shape[0])) for i in range(N)]
        steps = 0
        total_reward = 0
        done = [False] * N
        while not done[0] and steps < max_steps:
            steps += 1
            action_list = [agent_list[i].act(cur_state_list[i]) for i in range(N)]
            new_state_list, reward_list, done_list, _ = env.step(action_list)
            new_state_list = [new_state_list[i].reshape(1,env.observation_space[i].shape[0]) for i in range(N)]
            for i in range(N):
                agent_list[i].update_model(cur_state_list[i],action_list[i],reward_list[i],new_state_list[i],done[i])

            cur_state_list = new_state_list
            total_reward += np.sum(reward_list)
            if episode % render_every == 0:
               env.render()

        total_reward_list.append(total_reward)
        episode_length_list.append(steps)
        print('episode {} steps: {}, total reward: {},  elapsed time: {}s'.format(episode, steps, total_reward, int(time.time()-start_time)))

    plot_rewards_and_length(total_reward_list, episode_length_list)

if __name__ == "__main__":
    main()
