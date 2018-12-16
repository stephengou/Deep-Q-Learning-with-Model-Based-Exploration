import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd

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

def plot_state_scatter(agent,title1,title2,xlabel1,ylabel1,xlabel2,ylabel2,color, lim1 = [-0.1,0.1,-1.4,0.6],lim2=[-2.0,1.0,-2.0,2.0]):
    fig = plt.figure()

    a = []
    b = []
    sample_size = min(2000,len(agent.replay_memory))
    for sample in random.sample(agent.replay_memory, sample_size):
        a.append(sample[0][0][0])
        b.append(sample[0][0][1])

    sub1 = fig.add_subplot(2,2,1)
    sub1.grid(True,linewidth='0.4',color='white')
    sub1.set_xlabel(xlabel1)
    sub1.set_ylabel(ylabel1)
    sub1.set_ylim(bottom=lim1[0],top = lim1[1])
    sub1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    sub1.set_xlim(left=lim1[2],right=lim1[3])
    sub1.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    sub1.set_facecolor('#e6f3ff')
    sub1.scatter(a,b,s=3,color = color)

    if len(sample[0][0]) <= 2:
        return
    c = []
    d = []
    for sample in random.sample(agent.replay_memory, sample_size):
        c.append(sample[0][0][2])
        d.append(sample[0][0][3])

    sub2 = fig.add_subplot(2,2,2)
    sub2.grid(True,linewidth='0.4',color='white')
    sub2.set_xlabel(xlabel2)
    sub2.set_ylabel(ylabel2)
    sub2.set_ylim(bottom=lim2[0],top = lim2[1])
    sub2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    sub2.set_xlim(left=lim2[2],right=lim2[3])
    sub2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    sub2.set_facecolor('#e6f3ff')
    sub2.scatter(c,d,s=3,color = color)


def plot_rewards_and_length(rewards, min_reward,max_reward, lengths):

    rewards_df = pd.DataFrame(rewards)
    rewards_df.to_csv('Data/rewards.csv')

    fig = plt.figure()
    sub1 = fig.add_subplot(2,2,1)
    sub1.set_title('Reward')
    sub1.set_ylim(bottom=min_reward,top=max_reward)
    sub1.set_xlabel('episodes')
    sub1.set_ylabel('reward')
    sub1.plot(rewards)

    '''
    sub2 = fig.add_subplot(2,2,2)
    sub2.set_title('episode length')
    sub2.set_xlabel('episodes')
    sub2.plot(lengths)
    '''


    avg_reward = [0.] * len(rewards)
    cumulative_rewards = [0.] * len(rewards)
    cumulated_r = 0.
    for i in range(len(rewards)):
        cumulated_r += rewards[i]
        cumulative_rewards[i] = cumulated_r
    #interval = 10

    for i in range(len(rewards)):
        if i <= 0:
            avg_reward[i] = rewards[i]
        else:
            avg_reward[i] = (cumulative_rewards[i] - cumulative_rewards[0])/i
    sub3 = fig.add_subplot(2,2,2)
    sub3.set_ylim(bottom=min_reward,top=max_reward)
    sub3.set_title('average rewards')
    sub3.set_xlabel('episodes')
    sub3.plot(avg_reward)
    plt.show()



