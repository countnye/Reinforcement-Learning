import math
import numpy as np
import matplotlib.pyplot as plt
import bandit as bb


class Environment:
    def __init__(self, t, n, k, bandit_type):
        # define the number of iterations.
        self.T = t
        # define the number of bandits
        self.N = n
        # create N bandits with k arms of given type
        if bandit_type == 'b':
            self.bandits = [bb.Bandit(k, bb.Type.BERNOULLI) for _ in range(self.N)]
        elif bandit_type == 'g':
            self.bandits = [bb.Bandit(k, bb.Type.GAUSSIAN) for _ in range(self.N)]

    # agent picks random arms (to check agent implementation)
    def random_strategy(self):
        for _ in range(self.T):
            for idx, bandit in enumerate(self.bandits):
                chosen_arm = np.random.choice(bandit.k)
                bandit.chooseArm(chosen_arm)
                print('Bandit ', idx, ' chose arm ', chosen_arm)

    # function for greedy strategy
    # negative reward is the only case where the initial arm is non-negative
    def greedy(self):
        for t in range(self.T):
            for bandit in self.bandits:
                # for every arm, calculate the action values
                action_value = bandit.q_t()
                # if first iteration, pick a random arm
                zero_count = 0
                for val in action_value:
                    if val == 0.0:
                        zero_count += 1
                if zero_count == bandit.k:
                    chosen_arm = np.random.randint(0, bandit.k)
                else:
                    # select the arm with the highest action value
                    chosen_arm = action_value.index(max(action_value))
                # execute the chosen action
                bandit.chooseArm(chosen_arm)
                # update the best arm probability for each iteration
                bandit.update_best_arm_prob()
                # update regret for each iteration
                bandit.update_regret(t)

    # function for epsilon greedy strategy
    def e_greedy(self, e):
        for t in range(self.T):
            for bandit in self.bandits:
                # explore e% of iterations
                if np.random.random() > e:
                    chosen_arm = np.random.randint(0, bandit.k)
                # exploit (1 - e)% of iterations
                else:
                    # for every arm, calculate the action value
                    action_value = bandit.q_t()
                    zero_count = 0
                    for val in action_value:
                        if val == 0.0:
                            zero_count += 1
                    # if on first iteration, randomly choose an arm
                    if zero_count == bandit.k:
                        chosen_arm = np.random.randint(0, bandit.k)
                    else:
                        # select the arm with the highest action value
                        chosen_arm = action_value.index(max(action_value))
                    # execute the chosen action
                bandit.chooseArm(chosen_arm)
                # update the best arm probability for each iteration
                bandit.update_best_arm_prob()
                # update regret for each iteration
                bandit.update_regret(t)

    # function for optimistic initial values strategy
    def optimistic(self):
        # starts by assigning all actions an initial value greater than
        # the mean reward we expect to receive after pulling each arm
        for t in range(self.T):
            # initialise high action values for all bandits
            for bandit in self.bandits:
                bandit.initQ0(self.T)
            # execute greedy strategy for bandits
            for bandit in self.bandits:
                # for every arm, calculate the action value
                action_value = bandit.q_t()
                # select the arm with the highest action value
                chosen_arm = action_value.index(max(action_value))
                # execute the chosen action
                reward = bandit.chooseArm(chosen_arm)
                # replace reward estimate with obtained reward
                bandit.action_value[chosen_arm] = reward
                # update the best arm probability for each iteration
                bandit.update_best_arm_prob()
                # update regret for each iteration
                bandit.update_regret(t)

    # function for UCB strategy
    def UCB(self, c):
        for t in range(1, self.T + 1):
            for bandit in self.bandits:
                q_t = bandit.q_t()
                # list of all actions and their calculated
                # confidence intervals
                a_t = [0.0 for _ in range(bandit.k)]
                for idx, action in enumerate(bandit.arm_count):
                    # if arm has not been used, set the second part of equation to a high value
                    if action == 0.0:
                        a_t[idx] = q_t[idx] + 10000
                    else:
                        a_t[idx] = q_t[idx] + c * (math.sqrt(math.log(t) / action))
                chosen_arm = a_t.index(max(a_t))
                # choose arm with largest UCB
                bandit.chooseArm(chosen_arm)
                # update the best arm probability for each iteration
                bandit.update_best_arm_prob()
                # update regret for each iteration
                bandit.update_regret(t)

    # function for action preferences strategy
    def action_preferences(self, alpha):
        H_t = None
        for t in range(1, self.T + 1):
            for bandit in self.bandits:
                # Initialise H_t with same value for all actions.
                # Only during first iteration
                if t == 1:
                    H_t = [10 for _ in range(bandit.k)]
                else:
                    # Compute policy at timestep
                    pi_t = (np.exp(H_t)) / (np.sum(np.exp(H_t)))
                    # choose an action based on prob. distr.
                    actions = np.arange(start=0, stop=bandit.k, step=1)
                    chosen_arm = np.random.choice(a=actions, p=pi_t)
                    # Get reward assocaited with action
                    r_t = bandit.chooseArm(chosen_arm)
                    avg_r = bandit.get_average_reward()
                    # Update H_t(a') based on chosen arm/action: a'
                    H_t[chosen_arm] = H_t[chosen_arm] + alpha * (r_t - avg_r) * (1 - pi_t[chosen_arm])
                    # Update H_t for all non-chosen arms
                    for _ in range(len(H_t)):
                        if _ == chosen_arm:
                            continue
                        else:
                            H_t[_] = H_t[_] - alpha * (r_t - avg_r) * (pi_t[_])

                    bandit.update_best_arm_prob()
                    bandit.update_regret(t)

                




    # function to plot percentage times the best arm was chosen
    def plot_best_arm_prob(self, strategy):
        x = [i for i in range(self.T)]
        y = [bandit.best_arm_prob for bandit in self.bandits]
        # plot the probabilities
        for num in range(self.N):
            plt.plot(x, y[num], label="bandit " + str(num))
        plt.title('Best Arm Probability for ' + strategy + ' strategy')
        plt.legend()
        plt.show()

    # function to plot regret over each iteration
    def plot_regret(self, bandit, strategy):
        x = [i for i in range(self.T)]
        y = [regret for regret in bandit.regret_over_t]
        # plot the probabilities
        for num in range(bandit.k):
            plt.plot(x, y[num], label="arm " + str(num))
        plt.title('Regret for arm using ' + strategy + ' strategy')
        plt.legend()
        plt.show()

    # function to print the stats
    def print_stats(self):
        print("FINAL RESULTS:")
        print('====================================')
        for num, agent in enumerate(self.bandits):
            print('Bandit ', num, "'s best arm = ", agent.best_arm)
            print('Bandit ', num, "'s arm count = ", agent.arm_count)
            print('Bandit ', num, "'s rewards = ", [round(item, 2) for item in agent.rewards])
            # how to get number of iteration? maybe global variable?
            print('Bandit ', num, "'s regret = ", [round(item, 2) for item in agent.get_regret(self.T)])
            print('====================================')


env = Environment(t=100, n=2, k=4, bandit_type='g')
env.action_preferences(0.1)
env.print_stats()

# (!1)
