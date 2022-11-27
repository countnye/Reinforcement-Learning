<<<<<<< HEAD
import numpy as np

import bernoulli_bandit as bb


class Environment:
    def __init__(self, t, n, k, bandit_type):
        # define number of iterations
        self.T = t
        # define the number of bandits
        self.N = n
        # create N bandits with k arms of given type
        if bandit_type == 'b':
            self.bandits = [bb.BernoulliBandit(k) for _ in range(self.N)]
        elif bandit_type == 'g':
            pass

    # agent picks random arms (to check agent implementation)
    def random_strategy(self):
        for _ in range(self.T):
            for idx, bandit in enumerate(self.bandits):
                chosen_arm = np.random.choice(bandit.k)
                bandit.chooseArm(chosen_arm)
                print('Bandit ', idx, ' chose arm ', chosen_arm)


# initialize the environment
env = Environment(10, 2, 3, 'b')
# execute the random strategy
env.random_strategy()
# print arm count and rewards for each bandit
print("FINAL RESULTS:")
for num, agent in enumerate(env.bandits):
    print('Bandit ', num, "'s arm count = ", agent.arm_count)
    print('Bandit ', num, "'s rewards = ", agent.rewards)


# (!1)
=======
from enum import Enum
import numpy as np

class bandit_type(Enum):
    GAUSSIAN = 0
    BERNOULLI = 1

class strategies(Enum):
    GREEDY = 0
    EPSILON = 1
    OPTIMISTIC = 2
    UCB = 3
    ACTION_PREF = 4

class bandit:
    def __init__(self, k, n, b_type):
        self.k = k
        self.n = n
        self.b_type = b_type
        self.t = 100
        self.mean_reward = 0
        self.strategy = strategies.GREEDY

    # Idk if bernoulli is 100% correct
    def get_reward(self):
        if self.b_type == bandit_type.GAUSSIAN:
            return np.random.normal()
        elif self.by_type == bandit_type.BERNOULLI:
            return np.random.binomial(size=1, n=1, p=0.1)

    def perform_action(self):
        # select action
        # get reward and update mean reward
        # WIP
        rd = self.get_reward()
        


    
        


    
    

>>>>>>> origin/assignment_1
