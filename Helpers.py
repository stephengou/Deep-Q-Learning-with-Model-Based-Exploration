import matplotlib.pyplot as plt

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
    plt.show()
