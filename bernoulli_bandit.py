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

    # function to return the rewards obtained by the bandit
    def getRewards(self):
        return self.rewards

# (!1)
