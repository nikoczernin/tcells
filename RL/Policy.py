import random

import numpy as np
from matplotlib import pyplot as plt

from RL import utils


class Policy():

    def __init__(self, actions):
        self.actions = actions
        self.n_actions = len(actions)

    def get_decision_probabilities(self, state, epsilon=None):
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()

    def pick_action(self, state, epsilon=None):
        # returns index of picked action
        decision_probs = self.get_decision_probabilities(state, epsilon)
        # print("decision_probs:", decision_probs)
        return self.actions[np.random.choice(self.n_actions, p=decision_probs)]


class LinearPolicy(Policy):
    def __init__(self, actions, n_features):
        super().__init__(actions)
        self.n_features = n_features
        # set initial linear coeffs
        self.W = np.zeros((self.n_actions, n_features))
        print(self)

    def q(self, s, index_a):
        """Linear action‐value Q(s,a)."""
        # index_a: index of the action a
        return np.dot(self.W[index_a], s)

    def get_decision_probabilities(self, state, epsilon=None):
        """Return a length‐n_actions probability vector."""
        # get an estimate of each action-value in the current state
        qs = np.array([self.q(state, a) for a in range(self.n_actions)])
        # if there is no epislon given, use the action-values as decision probabilities
        if epsilon is None:
            qs = utils.stable_softmax(qs)
            return qs
        else:
            # with probability epsilon, explore and use a random action
            if random.random() < epsilon:
                return np.ones(self.n_actions) / self.n_actions
            # else pick the maximum best action-value action
            else:
                # greedy: 1 for max‐arg, 0 elsewhere (broken ties arbitrarily)
                probs = np.zeros_like(qs)
                probs[np.argmax(qs)] = 1.0
                return probs

    def __str__(self):
        out = ""
        for action_index, action in enumerate(self.actions):
            out += f"{str(action)}: {self.W[action_index]}\n"
        # return f"LinearPolicy\n{self.W}"
        return out

    def plot(self, agent):
        # Y = [self.env.get_certainty(t) for t in X]
        X = np.arange(agent.T)
        for a in range(len(agent.env.actions)):
            label = agent.env.actions[a]
            # plt.plot(X, Y)
            Y = [agent.policy.q(np.array([t, agent.env.get_certainty(t), 0]), a) for t in X]
            plt.plot(X, Y, label=label)
        plt.xlabel("search time")
        plt.ylabel("action-value")
        plt.title("Action-values per action over search time")
        plt.grid(True)
        plt.legend()
        plt.show()


class APCThresholdPolicy(Policy):
    def __init__(self, actions, threshold, T):
        super().__init__(actions)
        self.T = T # max time steps
        self.threshold = threshold

    def get_decision_probabilities(self, state, epsilon=None):
        """
        returns: if certainty > threshold,
        return random choice of "call" or "skip", else return "stay"
        """
        t = state[0]
        certainty = state[1]
        if certainty > self.threshold or t == self.T-1:
            return random.choice([
                [0, 1, 0],
                [0, 0, 1]
            ])
        else:
            return np.array([1, 0, 0])

    def __str__(self):
        return f"APCThresholdPolicy threshold={self.threshold}"

    def plot(self, tcell):
        X = np.arange(tcell.T)
        Y = [tcell.env.get_certainty(t) for t in X]
        plt.plot(X, Y)
        plt.xlabel("search time")
        plt.ylabel("certainty")
        plt.title("Decision certainty over search time")
        plt.axhline(y=self.threshold, color='r', linestyle='-')
        plt.grid(True)
        plt.legend()
        plt.show()