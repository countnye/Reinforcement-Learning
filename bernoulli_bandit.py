import numpy as np


# define the Bernoulli bandit
class BernoulliBandit:
    def __init__(self, k):
        # define the number of arms/actions for the bandit
        self.k = k
        # define an array to store all the rewards obtained
        self.rewards = [0 for _ in range(self.k)]
        # pick k random probability values for Bernoulli distribution
        self.prob = []
        for _ in range(self.k):
            self.prob.append(round(np.random.uniform(0, 1), 2))
        # store the number of times an action has been taken
        self.arm_count = [0 for _ in range(self.k)]

    # function to get reward for k arms/actions for all the bandits
    def chooseArm(self, a):
        # if chosen value is higher than win prob for given arm, reward is 1
        val = np.random.random()
        if val > self.prob[a]:
            self.rewards[a] += 1
        # increment arm count by 1
        self.arm_count[a] += 1

    # function to get the current action value estimate
    def q_t(self):
        action_value = [0.0 for _ in range(self.k)]
        for a in range(self.k):
            # if arm has not been used, no reward is possible
            if self.arm_count[a] == 0:
                action_value[a] = 0.0
            else:
                # action value is sum of rewards/action count for a given action
                action_value[a] = self.rewards[a]/self.arm_count[a]
        return action_value

    # function to return the rewards obtained by the bandit
    def getRewards(self):
        return self.rewards

    # function to calculate max reward possible, used to calculate regret
    def getMaxReward(self, n):
        max_reward = [0 for _ in range(self.k)]
        # for each arm, the max reward is simply getting 1 for each iteration
        for idx in range(self.k):
            for _ in range(n):
                max_reward[idx] += 1
        return max_reward

    # function to calculate regret
    def getRegret(self, n):
        regret = []
        max_reward = self.getMaxReward(n)
        curr_reward = self.getRewards()
        for i in range(self.k):
            regret.append(max_reward[i] - curr_reward[i])
        return regret

# (!1)
