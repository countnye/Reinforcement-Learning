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

    # function for greedy strategy
    def greedy(self):
        for _ in range(self.T):
            for bandit in self.bandits:
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


# initialize the environment
env = Environment(10, 2, 3, 'b')
# execute the random strategy
env.greedy()
# print arm count and rewards for each bandit
print("FINAL RESULTS:")
for num, agent in enumerate(env.bandits):
    print('Bandit ', num, "'s arm count = ", agent.arm_count)
    print('Bandit ', num, "'s rewards = ", agent.rewards)
    # how to get number of iteration? maybe global variable?
    print('Bandit ', num, "'s regret = ", agent.getRegret(10))
    print('====================================')

# (!1)
