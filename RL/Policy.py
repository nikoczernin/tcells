import random
import copy
import numpy as np
from matplotlib import pyplot as plt

from RL import utils


class Policy():

    def __init__(self, actions, verbose=False):
        self.actions = actions
        self.verbose = verbose
        self.n_actions = len(actions)

    def get_decision_probabilities(self, state, epsilon=None):
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()

    def pick_action(self, state, epsilon=None):
        # returns index of picked action
        decision_probs = self.get_decision_probabilities(state, epsilon)
        # print(self.actions, decision_probs)
        # print(self.n_actions, p=decision_probs)
        # print("decision_probs:", decision_probs)
        return self.actions[np.random.choice(self.n_actions, p=decision_probs)]


class LinearPolicy(Policy):
    def __init__(self, actions, n_features):
        raise NotImplementedError("Starting working here if you ever care!")
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
        for a in range(len(agent.env.ACTIONS)):
            label = agent.env.ACTIONS[a]
            # plt.plot(X, Y)
            Y = [agent.policy.q(np.array([t, agent.env.get_certainty(t), 0]), a) for t in X]
            plt.plot(X, Y, label=label)
        plt.xlabel("search time")
        plt.ylabel("action-value")
        plt.title("Action-values per action over search time")
        plt.grid(True)
        # plt.legend()
        plt.show()


class SingleSearchPhasePolicy(Policy):
    # stop the exploration of the APC at hand at a fixed point tau in T
    # when t == tau, stop and make a classification
    def __init__(self, actions, tau, verbose=False):
        super().__init__(actions, verbose=verbose)
        if tau is not None:
            if tau < 1:
                ValueError("Stopping point tau must be >= 1. How you gonna have negative time bruh?")
        self.tau = tau # stopping time

    def get_decision_probabilities(self, state, epsilon=None):
        """
        Params:
        state: expected format [timestep t:int > 0, evidence e: float in [0,1], terminate: bool]
        epsilon: float in [0,1]

        returns: if timestep t >= tau, tell the TCell in the APC to make a decision
        return probabilities of picking "stay" or "positive" or "negative"
        probabilities are hard-lined to 0 and 1
        """
        t = int(state[0])
        certainty = state[1] # probability of APC being positive
        c_ = round(certainty, 4)
        # if the evidence reaches the stopping point tau...
        if t == self.tau:
            # simulate a T-Cell decision
            # sample from uniform distribution and compare to certainty
            if random.random() < certainty:
                if self.verbose: print(t, c_, "\tClassify positive!")
                return np.array([0, 1, 0])
            else:
                if self.verbose: print(t, c_, "\tClassify negative!")
                return np.array([0, 0, 1])
        # otherwise definitely take action "stay"
        else:
            if self.verbose: print(t, c_, "\tExplore a little longer!")
            return np.array([1, 0, 0])

    def __str__(self):
        return f"Stopping point=({self.tau})"


class DoubleSearchPhasePolicy(SingleSearchPhasePolicy):
    """
    DoubleSearchPhasePolicy is an implementation of a 2-phase search strategy
    there are two stopping points: tau_1 < tau_2
    as certainty c(t) converges to 1 or 0, eventually tau_1 is reached
    at this point, the singe phase search strategy would make a final decision
    the first, liberally early stopping point would lead this decision to be based on very
    little certainty, more of which would be required to be likely correct about the decision
    instead of a making a final decision at the first stopping point,
    premeditate a random decision of either positive or negative
    if that decision is positive, keep going and collect more evidence, until you reach stopping point tau_2
    at this point, more evidence was collected and any next decision would lead to a lower false positive rate
    whatever decision is made now is final
    in the opposite case of certainty converging to 0 instead of 1,
    think the same way, not vice versa: after reaching tau_1, make a random decision (dep on c(tau_1))
    and when its positive, keep going until tau_2, or, if its negative, terminate
    """
    def __init__(self, actions, tau_1:int, tau_2:int, verbose=False):
        """
        :param actions:
        :param tau_1: first stopping point >= 1
        :param tau_2: second stopping point >= 2
        """
        if tau_1 < 1 or tau_2 < 2:
            raise ValueError("tau_1 must be at least 1 and tau_2 must be at least 2")
        if tau_1 >= tau_2:
            raise ValueError("tau_1 must be lower than tau_2")
        super().__init__(actions, None, verbose=verbose)
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def get_decision_probabilities(self, state, epsilon=None):
        """
        Params:
        state: expected format [timestep t:int > 0, evidence e: float in [0,1], terminate: bool]
        epsilon: float in [0,1]

        returns: if timestep t >= tau, tell the TCell in the APC to make a decision
        return probabilities of picking "stay" or "classify"
        probabilities are hard-lined to 0 and 1
        """
        t = state[0]
        c = state[1] # probability of APC being positive
        # if the evidence reaches the FIRST stopping point tau_1...
        c_ = round(c, 4) # rounded c for pretty printing
        if t != self.tau_1 and t != self.tau_2:
            if self.verbose: print(t, c_, "\tToo soon! Explore a little longer!")
            return np.array([1, 0, 0])

        # simulate a decision using the given certainty
        positive_decision = random.random() < c
        # if we are at the first stopping point and the decision was positive
        # postpone and keep exploring
        if t == self.tau_1 and positive_decision:
            if self.verbose: print(t, c_, "\tIt could be positive, but look deeper!")
            return np.array([1, 0, 0])
        # else, we are classifying!
        else:
            if self.verbose: print(f"Classifying {'positive' if positive_decision else 'negative'}")
            return np.array([0, int(positive_decision), int(not positive_decision)])


    def __str__(self):
        return f"Stopping points=({self.tau_1, self.tau_2})"












