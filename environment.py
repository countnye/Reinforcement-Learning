import numpy as np

import bandit as bb


class Environment:
    def __init__(self, time, t, n, k, bandit_type):
        # define the number of iterations.
        # Is also the time
        self.time = time
        # define the number of bandits
        self.N = n
        # define the number of times the 
        # experiment should be run for
        self.T = t
        # create N bandits with k arms of given type
        if bandit_type == 'b':
            self.bandits = [bb.Bandit(k, bb.Type.BERNOULLI) for _ in range(self.N)]
        elif bandit_type == 'g':
            self.bandits = [bb.Bandit(k, bb.Type.GAUSSIAN) for _ in range(self.N)]

    # agent picks random arms (to check agent implementation)
    def random_strategy(self):
        for _ in range(self.time):
            for idx, bandit in enumerate(self.bandits):
                chosen_arm = np.random.choice(bandit.k)
                bandit.chooseArm(chosen_arm)
                print('Bandit ', idx, ' chose arm ', chosen_arm)

    # function for greedy strategy
    def greedy(self):
        for _ in range(self.time):
            for bandit in self.bandits:
                # for every arm, calculate the action value
                action_value = bandit.q_t()
                # select the arm with the highest action value
                chosen_arm = action_value.index(max(action_value))
                # execute the chosen action
                bandit.chooseArm(chosen_arm)

    # function for epsilon greedy strategy
    def e_greedy(self, e):
        for _ in range(self.time):
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
        for _ in range(self.time):
            # Initialise high action values for all bandits
            # NEED TO IMPLEMENT initQ0()
            map(lambda x: x.initQ0(), self.bandits)
            # Execute greedy strategy for bandits
            for bandit in self.bandits:
                # for every arm, calculate the action value
                action_value = bandit.q_t()
                # select the arm with the highest action value
                chosen_arm = action_value.index(max(action_value))
                # execute the chosen action
                bandit.chooseArm(chosen_arm)

    # function for UCB strategy
    def UCB(self, c):
        for t in range(1, self.time):
            for bandit in self.bandits:
                q_t = bandit.q_t()
                a_t = []
                # For each action, calculate a_t
                # n_a: number of times all actions were selected
                for action in range(bandit.k):
                    n_a = bandit.n_a(action)
                    # list of all actions and their calculated
                    # confidence intervals
                    a_t.append(q_t[action] + c * (np.sqrt(np.log(t) / n_a)))
                # choose arm with largest UCB
                print("Q_T: ", q_t, "A_T: ", a_t)
                chosen_arm = a_t.index(max(a_t))
                print("chosen arm: ", chosen_arm)
                bandit.chooseArm(chosen_arm)

    # function for action preferences strategy
    def action_preferences(self):
        pass


# initialize the environment
env = Environment(time=10, n=2, t=100, k=3, bandit_type='g')
# execute the random strategy
# env.UCB(0.1)
env.e_greedy(0.1)
# print arm count and rewards for each bandit
print("FINAL RESULTS:")
for num, agent in enumerate(env.bandits):
    print('Bandit ', num, "'s arm count = ", agent.arm_count)
    print('Bandit ', num, "'s rewards = ", agent.rewards)
    # how to get number of iteration? maybe global variable?
    print('Bandit ', num, "'s regret = ", agent.get_regret(10))
    print('====================================')

# (!1)
