import environment as env

# initialize the environment
state = env.Environment(t=10, n=3, k=4, bandit_type='g')
# execute the chosen strategy
state.UCB(0.1)
# print the final stats
state.print_stats()
# plot the best arm probability
# state.plot_best_arm_prob('UCB')
# plot the regret of each arm for each bandit
for bandit in state.bandits:
    state.plot_regret(bandit, 'UCB')

# (!1)
