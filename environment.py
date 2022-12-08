import math
import numpy as np
import matplotlib.pyplot as plt
import bandit as bb


class Environment:
    def __init__(self, epochs, t, n, k, bandit_type):
        # define the number of epochs
        self.epochs = epochs
        # define the number of iterations per epoch
        self.T = t
        # define the number of bandits
        self.N = n
        # create N bandits with k arms of given type
        if bandit_type == 'b':
            self.bandits = [bb.Bandit(k, bb.Type.BERNOULLI) for _ in range(self.N)]
        elif bandit_type == 'g':
            self.bandits = [bb.Bandit(k, bb.Type.GAUSSIAN) for _ in range(self.N)]
        # store the average reward for each run per epoch for each bandit
        self.epoch_reward = []
        self.epoch_best_arm = []
        # store the average reward obtained by each arm for each bandit
        self.arm_reward = [[0.0 for _ in range(k)] for _ in range(self.N)]
        # store the average arm count for each arm for each bandit
        self.arm_count = [[0.0 for _ in range(k)] for _ in range(self.N)]

    # function for greedy strategy
    def greedy(self):
        run_reward = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        best_arm_prob = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        for _ in range(self.epochs):
            # reset the action values
            for b in self.bandits:
                b.reset_bandit()
            for t in range(self.T):
                for idx, bandit in enumerate(self.bandits):
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
                    reward = bandit.chooseArm(chosen_arm)
                    # increment reward for chosen arm
                    self.arm_reward[idx][chosen_arm] += reward
                    # increment arm count for chosen arm
                    self.arm_count[idx][chosen_arm] += 1
                    # update regret for each iteration
                    bandit.update_regret(t)
                    # update the best arm probability for each iteration
                    best_arm_prob[idx][t] += bandit.best_arm_prob()
                    # add the reward to the run corresponding to the bandit
                    run_reward[idx][t] += reward
        # average the run reward
        for i in range(self.N):
            for j in range(self.T):
                run_reward[i][j] /= self.epochs
                best_arm_prob[i][j] /= self.epochs
            self.epoch_reward.append(run_reward[i])
            self.epoch_best_arm.append(best_arm_prob[i])

    # function for epsilon greedy strategy
    def e_greedy(self, e):
        run_reward = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        best_arm_prob = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        for _ in range(self.epochs):
            # reset the action values
            for b in self.bandits:
                b.reset_bandit()
            for t in range(self.T):
                for idx, bandit in enumerate(self.bandits):
                    # explore e% of iterations
                    if np.random.random() < e:
                        chosen_arm = np.random.randint(0, bandit.k)
                    # exploit (1 - e)% of iterations
                    else:
                        # for every arm, calculate the action value
                        action_value = bandit.q_t()
                        # select the arm with the highest action value
                        chosen_arm = action_value.index(max(action_value))
                        # execute the chosen action
                    reward = bandit.chooseArm(chosen_arm)
                    # increment reward for chosen arm
                    self.arm_reward[idx][chosen_arm] += reward
                    # increment arm count for chosen arm
                    self.arm_count[idx][chosen_arm] += 1
                    # update the best arm probability for each iteration
                    best_arm_prob[idx][t] += bandit.best_arm_prob()
                    # add the reward to the run corresponding to the bandit
                    run_reward[idx][t] += reward
        # average the run reward
        for i in range(self.N):
            for j in range(self.T):
                run_reward[i][j] /= self.epochs
                best_arm_prob[i][j] /= self.epochs
            self.epoch_reward.append(run_reward[i])
            self.epoch_best_arm.append(best_arm_prob[i])

    # function for optimistic initial values strategy
    def optimistic(self):
        run_reward = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        best_arm_prob = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        # starts by assigning all actions an initial value greater than
        # the mean reward we expect to receive after pulling each arm
        for _ in range(self.epochs):
            # reset the action values
            for b in self.bandits:
                b.reset_bandit()
            for t in range(self.T):
                # initialise high action values for all bandits
                for bandit in self.bandits:
                    bandit.initQ0(self.T)
                # execute greedy strategy for bandits
                for idx, bandit in enumerate(self.bandits):
                    # for every arm, calculate the action value
                    action_value = bandit.q_t()
                    # select the arm with the highest action value
                    chosen_arm = action_value.index(max(action_value))
                    # execute the chosen action
                    reward = bandit.chooseArm(chosen_arm)
                    # increment reward for chosen arm
                    self.arm_reward[idx][chosen_arm] += reward
                    # increment arm count for chosen arm
                    self.arm_count[idx][chosen_arm] += 1
                    # replace reward estimate with obtained reward
                    bandit.action_value[chosen_arm] = reward
                    # update the best arm probability for each iteration
                    best_arm_prob[idx][t] += bandit.best_arm_prob()
                    # add the reward to the run corresponding to the bandit
                    run_reward[idx][t] += reward
        # average the run reward
        for i in range(self.N):
            for j in range(self.T):
                run_reward[i][j] /= self.epochs
                best_arm_prob[i][j] /= self.epochs
            self.epoch_reward.append(run_reward[i])
            self.epoch_best_arm.append(best_arm_prob[i])

    # function for UCB strategy
    def UCB(self, c):
        run_reward = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        best_arm_prob = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        for _ in range(self.epochs):
            # reset the action values
            for b in self.bandits:
                b.reset_bandit()
            for t in range(1, self.T + 1):
                for idx, bandit in enumerate(self.bandits):
                    q_t = bandit.q_t()
                    # list of all actions and their calculated
                    # confidence intervals
                    a_t = [0.0 for _ in range(bandit.k)]
                    for index, action in enumerate(bandit.arm_count):
                        # if arm has not been used, set the second part of equation to a high value
                        if action == 0:
                            a_t[index] = q_t[index] + 2 * bandit.k
                        else:
                            a_t[index] = q_t[index] + c * (math.sqrt(math.log(t) / action))
                    chosen_arm = a_t.index(max(a_t))
                    # choose arm with largest UCB
                    reward = bandit.chooseArm(chosen_arm)
                    # increment reward for chosen arm
                    self.arm_reward[idx][chosen_arm] += reward
                    # increment arm count for chosen arm
                    self.arm_count[idx][chosen_arm] += 1
                    # update the best arm probability for each iteration
                    best_arm_prob[idx][t - 1] += bandit.best_arm_prob()
                    # add the reward to the run corresponding to the bandit
                    run_reward[idx][t - 1] += reward
        # average the run reward
        for i in range(self.N):
            for j in range(self.T):
                run_reward[i][j] /= self.epochs
                best_arm_prob[i][j] /= self.epochs
            self.epoch_reward.append(run_reward[i])
            self.epoch_best_arm.append(best_arm_prob[i])

    # function for action preferences strategy
    def action_preferences(self, alpha):
        H_t = None
        run_reward = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        best_arm_prob = [[0.0 for _ in range(self.T)] for _ in range(self.N)]
        for _ in range(self.epochs):
            # reset the action values
            for b in self.bandits:
                b.reset_bandit()
            for t in range(1, self.T + 1):
                for idx, bandit in enumerate(self.bandits):
                    # initialise H_t with same value for all actions.
                    # only during first iteration
                    if t == 1:
                        H_t = [0 for _ in range(bandit.k)]
                    else:
                        # compute policy at timestep
                        pi_t = (np.exp(H_t)) / (np.sum(np.exp(H_t)))
                        # choose an action based on probability distribution
                        actions = np.arange(start=0, stop=bandit.k, step=1)
                        chosen_arm = np.random.choice(a=actions, p=pi_t)
                        # get reward associated with action
                        r_t = bandit.chooseArm(chosen_arm)
                        # increment reward for chosen arm
                        self.arm_reward[idx][chosen_arm] += r_t
                        # increment arm count for chosen arm
                        self.arm_count[idx][chosen_arm] += 1
                        avg_r = bandit.get_average_reward(t)
                        # update H_t for all non-chosen arms
                        for _ in range(len(H_t)):
                            if _ == chosen_arm:
                                # update H_t(a') based on chosen arm/action: a'
                                H_t[chosen_arm] = H_t[chosen_arm] + alpha * (r_t - avg_r) * (1 - pi_t[chosen_arm])
                                continue
                            else:
                                H_t[_] = H_t[_] - alpha * (r_t - avg_r) * (pi_t[_])
                        bandit.update_regret(t)
                        # update the best arm probability for each iteration
                        best_arm_prob[idx][t - 1] += bandit.best_arm_prob()
                        # add the reward to the run corresponding to the bandit
                        run_reward[idx][t - 1] += r_t
        # average the run reward
        for i in range(self.N):
            for j in range(self.T):
                run_reward[i][j] /= self.epochs
                best_arm_prob[i][j] /= self.epochs
            self.epoch_reward.append(run_reward[i])
            self.epoch_best_arm.append(best_arm_prob[i])

    # function to plot percentage times the best arm was chosen
    def plot_best_arm_prob(self, strategy):
        x = [i for i in range(self.T)]
        y = self.epoch_best_arm
        # plot the probabilities
        for num in range(self.N):
            plt.plot(x, y[num], label="bandit " + str(num))
        plt.title('Best Arm Probability for ' + strategy + ' strategy')
        plt.legend()
        plt.show()

    # function to plot the average reward over each arm for each strategy
    def plot_reward(self, strategy):
        x = [i for i in range(self.T)]
        fig, ax = plt.subplots()
        # plot the reward
        for num in range(self.N):
            y = self.epoch_reward[num]
            ax.plot(x, y, label="bandit " + str(num))
        plt.title('Average reward using ' + strategy + ' strategy')
        plt.xlabel('Iteration number')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()

    # function to reset the bandit
    def reset(self):
        for bandit in self.bandits:
            bandit.epoch_reset()
        self.epoch_reward = []
        self.epoch_best_arm = []

    # function to write average reward and best arm probability to a singular list
    def write_to_list(self, list1, list2):
        list1.append(self.epoch_reward)
        list2.append(self.epoch_best_arm)
