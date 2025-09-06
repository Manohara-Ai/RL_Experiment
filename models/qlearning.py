import numpy as np

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

class EpsilonGreedy:
    def __init__(self, epsilon, rng):
        self.epsilon = epsilon
        self.rng = rng

    def choose_action(self, action_space, state, qtable):
        if self.rng.uniform(0, 1) < self.epsilon:
            return action_space.sample()
        else:
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            return self.rng.choice(max_ids)