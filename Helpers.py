import matplotlib.pyplot as plt
import random

def make_multi_env(scenario_name, benchmark=False,done_cb=None):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,done_callback=done_cb)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,done_callback=done_cb)
    return env

def plot_state_scatter(agent):
    a = []
    b = []
    for sample in random.sample(agent.replay_memory, 2000):
        a.append(sample[0][0][0])
        b.append(sample[0][0][1])
    plt.scatter(a,b)
def plot_rewards_and_length(rewards,lengths):
    fig = plt.figure()
    sub1 = fig.add_subplot(2,2,1)
    sub1.set_title('reward')
    sub1.set_xlabel('episodes')
    sub1.plot(rewards)
    sub2 = fig.add_subplot(2,2,2)
    sub2.set_title('episode length')
    sub2.set_xlabel('episodes')
    sub2.plot(lengths)

    avg_reward = [0.] * len(rewards)
    cumulative_rewards = [0.] * len(rewards)
    cumulated_r = 0.
    for i in range(len(rewards)):
        cumulated_r += rewards[i]
        cumulative_rewards[i] = cumulated_r
    interval = 10

    for i in range(len(rewards)):
        if i - interval < 0:
            avg_reward[i] = rewards[i]
        else:
            avg_reward[i] = (cumulative_rewards[i] - cumulative_rewards[i - interval])/interval
    sub3 = fig.add_subplot(2,2,3)
    sub3.set_title('average rewards')
    sub3.set_xlabel('episodes')
    sub3.plot(avg_reward)
    plt.show()
