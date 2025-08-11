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
        # plt.legend()
        plt.show()


class APCThresholdPolicy(Policy):
    def __init__(self, actions, threshold, T):
        super().__init__(actions)
        self.T = T # max time steps
        # this threshold is the minimum amount of evidence e(t) required
        # to make a decision
        self.threshold = threshold

    def get_decision_probabilities(self, state, epsilon=None, hardline=True):
        """
        Params:
        state: expected format [timestep t:int > 0, evidence e: float in [0,1], terminate: bool]
        epsilon: float in [0,1]
        hardline: bool. If True, decision probabilities are always 0 or 1, otherwise proportional to evidence

        returns: if evidence e > threshold; or if e converges to 0 then e < (1-threshold),
        return probabilities of picking "call" or "skip", else return "stay"
        probabilities are hardlined to 0 and 1 by default but can be left at float level if desired
        """
        t = state[0]
        e = state[1] # probability of APC being positive
        if e > self.threshold or  e < (1-self.threshold) or t == self.T-1:
            if hardline:
                p, q = round(e), 1-round(e)
            else:
                p, q = e, 1-e
            return np.array([0, p, q])
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
        plt.grid(True)
        plt.legend()
        plt.show()


class APCDoubleThresholdPolicy(APCThresholdPolicy):
    """
    Double threshold is an implementation of a 2-phase search strategy
    there are two thresholds, the first is closer to 0.5 than the second
    as evidence e(t) converges to 1 or 0, eventually threshold 1 is passed
    at this point, the singe phase search strategy would make a final decision
    the first, very liberal threshold would lead this decision to be based on very
    little evidence, more of which would be required to be correct about the decision
    i.e. a threshold_1 of 0.75 would lead to a higher false positive rate than using a later threshold
    instead of a making a final decision at the first threshold, make a random decision of either positive or negative
    make the choice dependent on the given e(t), so evidence of 0.78 would likely lead to a positive decision
    then, only if the decision was negative, actually make the decision final and let the episode terminate
    if the first decision was positive, keep going and collect more evidence, until you reach threshold_2
    at this point, more evidence was collected and any next decision would lead to a lower false positive rate
    whatever decision is made now is final
    the same goes for the opposite case of evidence converging to 0 instead of 1, where making a negative decision
    at threshold_1 would lead to a higher false negative rate than waiting until you reach threshold_2
    making a positive decision at threshold_1 would lead to instant termination, whereas a negative decision
    lets the agent stick around until threshold 2
    """
    def __init__(self, actions, threshold_1:float, threshold_2:float, T:int):
        """
        :param actions:
        :param threshold_1: must be between 0.5 and 1
        :param threshold_2: must be higher than threshold_1
        :param T:
        """
        if threshold_1 < 0.5 or threshold_1 > 1:
            raise ValueError("threshold_1 must be between 0.5 and 1")
        if threshold_1 > threshold_2:
            raise ValueError("threshold_1 must be lower than threshold_2")
        super().__init__(actions, None, T)
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2


    def get_decision_probabilities(self, state, epsilon=None, hardline=None):
        """
        Params:
        state: expected format [timestep t:int > 0, evidence e: float in [0,1], terminate: bool]
        epsilon: float in [0,1]
        hardline: irrelevant

        returns: if evidence e > threshold; or if e converges to 0 then e < (1-threshold),
        return probabilities of picking "call" or "skip", else return "stay"
        """
        t = state[0]
        e = state[1] # probability of APC being positive
        # case 0: evidence is not enough for either treshold (only check threshold_1 because its smaller anyway)
        if (1 - self.threshold_1) < e < self.threshold_1:
            # pick decision: stay
            return np.array([1, 0, 0])
        # case 1: evidence is enough to pass threshold_1 but still lower than treshold_2
        if (self.threshold_1 < e < self.threshold_2) or ((1 - self.threshold_2) < e < (1 - self.threshold_1)):
            # use the evidence as probability of the decision to make here
            # e is the probability of making a positive decision in this first step
            positive_decision = random.random() < e
            # if the decision was predictable, i.e. decision is positive and e > 0.5, or negative decision and e < 0.5
            # keep exploring and thus continue to the second search phase
            if (positive_decision and e > 0.5) or (not positive_decision and e < 0.5):
                # pick decision: stay
                return np.array([1, 0, 0])
            # otherwise, make the unlikely and final decision
            else:
                p, q = round(e), 1-round(e)
                return np.array([0, p, q])
        # case 2: evidence is enough to pass threshold_2
        # case 3: also, if too much time passed and you reach the maximum time taken to make a decision, terminate
        elif e > self.threshold_2 or e < (1 - self.threshold_2) or t == self.T - 1:
            p, q = round(e), 1 - round(e)
            return np.array([0, p, q])
        raise NotImplementedError("We are reaching a situation we shouldn't be in! How can this line of code be reached?")