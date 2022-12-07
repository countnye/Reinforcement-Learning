import environment as env

# initialize the environment
state = env.Environment(epochs=10, t=1000, n=2, k=6, bandit_type='g')

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
state.plot_reward('E-Greedy')

# (!1)
