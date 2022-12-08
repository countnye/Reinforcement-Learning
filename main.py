import environment as env
import numpy as np
import matplotlib.pyplot as plt


# function to plot the different strategies together
def plot(param_list, string):
    strategies = ['GREEDY', 'E_GREEDY', 'OPTIMISTIC', 'UCB', 'ACTION_PREF']
    x_axis = [t for t in range(env.T)]
    fig, ax = plt.subplots()
    # for each of the 5 strategies plot REWARD
    for idx in range(5):
        ax.plot(x_axis, param_list[idx], label=strategies[idx])

    plt.title(string + " per strategy")
    plt.xlabel("Timestep")
    plt.ylabel(string)
    plt.legend()
    plt.show()


state = env.Environment(epochs=50, t=3000, n=1, k=5, bandit_type='g')

y_axis_reward = []
y_axis_arm = []

state.greedy()
state.write_to_list(y_axis_reward, y_axis_arm)
state.reset()

state.e_greedy(0.1)
state.write_to_list(y_axis_reward, y_axis_arm)
state.reset()

state.optimistic()
state.write_to_list(y_axis_reward, y_axis_arm)
state.reset()

state.UCB(1)
state.write_to_list(y_axis_reward, y_axis_arm)
state.reset()

state.action_preferences(0.1)
state.write_to_list(y_axis_reward, y_axis_arm)
state.reset()

y_axis_arm = np.squeeze(y_axis_arm)
y_axis_reward = np.squeeze(y_axis_reward)

plot(y_axis_arm, "% of choosing best arm")
plot(y_axis_reward, "Average Reward")
