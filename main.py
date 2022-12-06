import environment as env

# initialize the environment
state = env.Environment(epochs=1, t=1000, n=1, k=5, bandit_type='g')

# execute the chosen strategy
state.optimistic()

# print the final stats
state.print_stats()

# plot the best arm probability
# state.plot_best_arm_prob('E-Greedy')

# plot the regret of each arm for each bandit
# for bandit in state.bandits:
#     state.plot_regret(bandit, 'Greedy')

# plot the reward over time for each bandit
state.plot_reward('Optimistic')

# (!1)
