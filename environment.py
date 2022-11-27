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
