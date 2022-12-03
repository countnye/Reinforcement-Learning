import numpy as np
from enum import Enum
import random as rd


class Type(Enum):
    GAUSSIAN = 1
    BERNOULLI = 2


# initialize the bandit class
class Bandit:
    def __init__(self, k, bandit_type):
        # define the type of bandit (Gaussian/Bernoulli)
        self.type = bandit_type
        # define the number of arms/actions for the bandit
        self.k = k
        # define an array to store all the rewards obtained
        self.rewards = [0 for _ in range(self.k)]
        # stores the reward probabilities for bernoulli bandit and
        # the mean rewards for gaussian bandit
        self.reward_param = []
        # store the number of times an action has been taken
        self.arm_count = [0 for _ in range(self.k)]
        # store the best arm for the experiment run
        self.best_arm = 0
        # initialize bandit arms based on bandit type
        if self.type == Type.BERNOULLI:
            self.init_bernoulli_arms()
        elif self.type == Type.GAUSSIAN:
            self.init_gaussian_arms()
        # store the action value or reward estimate for each arm
        self.action_value = [0.0 for _ in range(self.k)]
        # store the probability of choosing best arm
        self.best_arm_prob = []
        # store the regret for each iteration
        self.regret_over_t = [[] for _ in range(self.k)]

    # initialise reward distributions for the Bernoulli bandit
    def init_bernoulli_arms(self):
        for _ in range(self.k):
            self.reward_param.append(round(np.random.uniform(0, 1), 2))
        print('Bernoulli arm probabilities are ', self.reward_param)
        self.best_arm = self.reward_param.index(max(self.reward_param))
        print('Best arm is ', self.best_arm)

    # initialize reward for the Gaussian bandit
    def init_gaussian_arms(self):
        # Generates and stores k different mean values to use with the
        # Gaussian arms. K random values are sampled without replacement
        self.reward_param = rd.sample(range(0, 100), self.k)
        print('Gaussian arm means are ', self.reward_param)
        self.best_arm = self.reward_param.index(max(self.reward_param))
        print('Best arm is ', self.best_arm)

    # function to set initial action values to a given value, used for optimistic strategy
    def initQ0(self, n):
        for idx, item in enumerate(self.action_value):
            # initialize the action values to twice the highest possible reward
            self.action_value[idx] = 2 * self.get_max_reward(n)[idx]

    # function to get reward for k arms/actions for all the bandits
    def chooseArm(self, a):
        reward = 0
        if self.type == Type.BERNOULLI:
            reward = self.bernoulli_reward(a)
        elif self.type == Type.GAUSSIAN:
            reward = self.gaussian_reward(a)
        return reward

    # Bernoulli reward function
    def bernoulli_reward(self, a):
        # if chosen value is higher than win prob for given arm, reward is 1
        val = np.random.random()
        if val > self.reward_param[a]:
            self.rewards[a] += 1
            self.arm_count[a] += 1
            return 1
        else:
            self.arm_count[a] += 1
            return 0

    # Gaussian reward function
    def gaussian_reward(self, a):
        # get reward sampled from gaussian distribution
        # with mean unique to the particular arm
        reward = np.random.normal(self.reward_param[a])
        self.rewards[a] += reward
        self.arm_count[a] += 1
        return reward

    # function to get the current action value estimate
    def q_t(self):
        for a in range(self.k):
            # if arm has not been used, no reward is possible
            if self.arm_count[a] == 0:
                continue
            else:
                # action value is sum of rewards/action count for a given action
                self.action_value[a] = self.rewards[a] / self.arm_count[a]
        return self.action_value

    # returns number of times all actions have been taken
    def n_a(self, a):
        return 1 if self.arm_count[a] < 1 else self.arm_count[a]

    # function to return the rewards obtained by the bandit
    def get_rewards(self):
        return self.rewards

    # returns average reward up to a timestep
    def get_average_reward(self):
        return sum(self.rewards) / len(self.rewards)

    # function to calculate max reward possible, used to calculate regret
    def get_max_reward(self, n):
        max_reward = [0 for _ in range(self.k)]
        if self.type == Type.BERNOULLI:
            # for each arm, max reward is simply getting 1 for each iteration
            for idx in range(self.k):
                for _ in range(n):
                    max_reward[idx] += 1
        elif self.type == Type.GAUSSIAN:
            # for each arm, max reward is getting the mean reward for each iteration
            highest_mean = max(self.reward_param)
            for idx in range(self.k):
                for _ in range(n):
                    max_reward[idx] += highest_mean
        return max_reward

    # function to calculate regret
    def get_regret(self, t):
        regret = []
        max_reward = self.get_max_reward(t)
        curr_reward = self.rewards
        for i in range(self.k):
            regret.append(max_reward[i] - curr_reward[i])
        return regret

    # function to update best arm probability
    def update_best_arm_prob(self):
        # print('best arm = ', self.best_arm)
        # print('best arm count = ', self.arm_count[self.best_arm])
        # print('total count = ', sum(self.arm_count))
        self.best_arm_prob.append(round((self.arm_count[self.best_arm]/sum(self.arm_count)) * 100, 2))

    # function to update the regret
    def update_regret(self, t):
        max_reward = self.get_max_reward(t)
        curr_reward = self.rewards
        for i in range(self.k):
            self.regret_over_t[i].append(max_reward[i] - curr_reward[i])

# (!1)
