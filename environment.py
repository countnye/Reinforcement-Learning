import math
import numpy as np
import bandit as bb

ITERATIONS = 1000


class Environment:
    def __init__(self, t, n, k, bandit_type):
        # define the number of iterations.
        # Is also the time
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
        for _ in range(self.T):
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

    # function for epsilon greedy strategy
    def e_greedy(self, e):
        for _ in range(self.T):
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

    # function for optimistic initial values strategy
    def optimistic(self):
        # starts by assigning all actions an initial value greater than
        # the mean reward we expect to receive after pulling each arm
        for _ in range(self.T):
            # Initialise high action values for all bandits
            for bandit in self.bandits:
                bandit.initQ0()
            # Execute greedy strategy for bandits
            for bandit in self.bandits:
                # for every arm, calculate the action value
                action_value = bandit.q_t()
                # select the arm with the highest action value
                chosen_arm = action_value.index(max(action_value))
                # execute the chosen action
                reward = bandit.chooseArm(chosen_arm)
                # replace reward estimate with obtained reward
                bandit.action_value[chosen_arm] = reward

    # function for UCB strategy
    def UCB(self, c):
        for t in range(1, self.T + 1):
            for bandit in self.bandits:
                q_t = bandit.q_t()
                # list of all actions and their calculated
                # confidence intervals
                a_t = [0.0 for _ in range(bandit.k)]
                for idx, action in enumerate(bandit.arm_count):
                    # fix this part
                    if action == 0.0:
                        action = 1
                    a_t[idx] = q_t[idx] + c * (math.sqrt(math.log(t) / action))
                chosen_arm = a_t.index(max(a_t))
                # choose arm with largest UCB
                bandit.chooseArm(chosen_arm)

    # function for action preferences strategy
    def action_preferences(self):
        # for t in range(self.T):
        #     for bandit in self.bandits:
        #         # if in first iteration, initial preferences are the reward parameters
        #         if t == 0:
        #             curr_preference = bandit.reward_param
        pass


# initialize the environment
env = Environment(t=ITERATIONS, n=2, k=3, bandit_type='g')
# execute the random strategy
env.optimistic()
# print arm count and rewards for each bandit
print("FINAL RESULTS:")
for num, agent in enumerate(env.bandits):
    print('Bandit ', num, "'s arm count = ", agent.arm_count)
    print('Bandit ', num, "'s rewards = ", agent.rewards)
    # how to get number of iteration? maybe global variable?
    print('Bandit ', num, "'s regret = ", agent.get_regret(ITERATIONS))
    print('====================================')

# (!1)
