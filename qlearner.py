from collections import defaultdict
import random
import numpy as np

class QLearner:
    def __init__(self, env, n_bins, alpha, discount_factor, max_epsilon, min_epsilon, decay, adaptive_mode=True, adaptive_binning=False, binning=None):
        self.env = env
        self.n_bins = n_bins
        self.bins = []
        self.alpha = alpha
        self.adaptive_mode = adaptive_mode
        self.discount_factor = discount_factor
        self.epsilon = max_epsilon
        self.epsilon_max = max_epsilon
        self.epsilon_min = min_epsilon
        self.epsilon_decay = decay
        self.visited = defaultdict(int)

        if adaptive_binning:
            self.bins = binning
        else:
            for low, high in zip(self.env.observation_space.low, self.env.observation_space.high):
                if np.isinf(low) or np.isinf(high):
                    low, high = -1, 1
                self.bins.append(np.linspace(low, high, self.n_bins-1))

        self.state_shape = tuple([self.n_bins] * self.env.observation_space.shape[0])
        self.action_shape = self.env.action_space.n
        self.Q = np.zeros(self.state_shape + (self.action_shape,))

    def discretize_state(self, obs):
        state = []
        for i in range(len(obs)):
            bin_idx = np.digitize(obs[i], self.bins[i])
            state.append(min(bin_idx, self.n_bins - 1))
        return tuple(state)

    def choose_action(self, state):
        rand_i = random.uniform(0, 1)
        if rand_i > self.epsilon:
            action = np.argmax(self.Q[state])
        else:
            action = self.env.action_space.sample()
        return action

    def update_Q(self, state, new_state, action, reward):
        self.visited[state] += 1
        if self.adaptive_mode:
            self.alpha = 60 / (59 + self.visited[state])

        self.Q[state][action] += self.alpha * (reward + self.discount_factor * np.max(self.Q[new_state]) - self.Q[state][action])

