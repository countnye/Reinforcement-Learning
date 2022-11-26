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
        


    
        


    
    

