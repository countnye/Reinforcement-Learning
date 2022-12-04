import environment as env

# initialize the environment
state = env.Environment(t=1000, n=3, k=4, bandit_type='g')

# execute the chosen strategy
state.e_greedy(0.3)

# print the final stats
state.print_stats()

# plot the best arm probability
# state.plot_best_arm_prob('UCB')

# plot the regret of each arm for each bandit
# for bandit in state.bandits:
#     state.plot_regret(bandit, 'Greedy')

# plot the reward over time for each bandit
state.plot_reward(state.bandits, 'Greedy')

# (!1)
