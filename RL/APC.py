# RL environment for a single Stochastic APC
# each timestep represents searching the APC more closely, which in turn costs time
# the longer an APC is investigated, the more certainty about whether it contains harmful antigen you should have
import copy
import random
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from RL.Environment import Environment
from RL import utils

##### States, Actions & Rewards #####
# States: Each state is an array of three values:
# 1. the timestep `t`, natural numbers
# 2. the certainty function = certainty about the APC being positive `c` $\in [0, 1]$.
    # This starts at `bias` and gets nudged in either direction toward 0 or 1, depending on the attribute `isPositive`.
# 3. `1` if this state is a terminal state, `0` at the start of an episode and made positive by the agent

# Hidden State:
# positive = True/False
# each APC has a probability of being harmful/positive (True) or benign/negative (False)
# this is hidden to an agent
# this gets assigned randomly at environment-creation or -reset

# Actions:
# - `stay`: advance to the next timestep without making a decision
# - `positive`: classify the APC to be `positive`
# - `negative`: classify the APC to be `negative`


# Rewards:
# stay: -1
# positive & pos (TP): 100
# positive & neg (FP): -100
# negative & pos (FN): -100
# negative & neg (TN): 100

def b(x, k=1):
    y = np.empty_like(x)
    mask = x <= 0.5
    y[mask] = (2*x[mask])**k / 2
    y[~mask] = 1 - (2*(1-x[~mask]))**k / 2
    return y

def get_biases(n):
    return np.linspace(0, 1, n + 2)[1: -1]


class StochasticAPC(Environment):
    # ACTIONS
    ACTIONS = ["stay", "positive", "negative"]
    # Params:
    # p: probability of the APC being positive/harmful
    # bias: starting point and general nudge upwards or downwards of the
    # certainty function. Default is 0.5, which is right in the middle,
    # in that case the certainty functions for positive and negative APCs are symmetrical
    # Any bias between 0.5 and 1 nudge the start and function upwards, leading to more
    # true positive and false positives, whereas a lower bias, between 0 and 0.5
    # will nudge the function downwards and lead to fewer true positives and false positives
    def __init__(self, certainty_fun=utils.rational_function,
                 p=0.05,
                 bias=0.5, # 0 <= bias <= 1
                 learning_rate:float=1.0,
                 positive=None):
        super().__init__(self.ACTIONS)
        # rewards
        self.rewards = {
            "stay": -1,
            "positive": {
                "TP": 100,
                "FP": -100,
            },
            "negative": {
                "TN": 100,
                "FN": -100,
            }
        }
        # probability of APC being positive
        # sets the starting point of the certainty function
        self.bias = None
        self.set_bias(bias)

        self.learning_rate = learning_rate * 0.5
        self.certainty_fun = certainty_fun
        # set the status
        self.p = p
        self.positive = positive
        if positive is None:
            self.reset()
        # starting_state
        self.starting_state = np.array([0., self.get_certainty(0), 0.])

    def reset(self):
        self.positive = random.random() < self.p

    def set_bias(self, bias):
        self.bias = bias / (1 - bias + (1e-12 if bias == 1 else 0))


    def state_is_terminal(self, state) -> bool:
        return state[2] == 1

    @staticmethod
    def print_state(state):
        print(f"State t={int(state[0])}, q={round(state[1], 4)}, stop={bool(state[2])}")

    def get_certainty(self, t):
        # returns a value between 0 and 1
        # for a positive APC the function converges monotonously to 1
        # for a negative APC the function converges monotonously to 0
        # at t=1, it is equal to the bias
        if t == 0: t = 1 # readjust the beginning
        # bias > 0.5: increased certainty, bias < 0.5: decreased certainty
        if self.positive:
            # adjust t by the bias
            # certainty = self.certainty_fun(t * self.bias, flatness=self.learning_rate)
            certainty =  t ** self.learning_rate * self.bias / (1 +  t ** self.learning_rate * self.bias)
        else:
            # adjust t by the inverted bias
            certainty =  1 - t ** self.learning_rate / self.bias / (1 +  t ** self.learning_rate / self.bias)
            # certainty = 1 -self.certainty_fun(t / self.bias, flatness=self.learning_rate)
        return certainty

    def apply_action(self, state, action):
        new_state = state.copy()
        # advance t (time)
        new_state[0] += 1
        # recompute certainty c (certainty of APC being positive)
        certainty = self.get_certainty(new_state[0])
        # print("Certainty=", certainty)
        new_state[1] = certainty
        # if ACTIONS are negative or positive, terminate
        if action == "positive" or action == "negative":
            new_state[2] = 1
        return new_state

    def get_reward(self, state: tuple, action: tuple, new_state: tuple = None):
        # state, action, new_state -> reward, event
        # if the agent wants to inspect the APC for longer
        # his action was "stay"
        # penalize him for the time it takes to investigate
        t = state[0]
        c = state[1]

        if action == "stay":
            return self.rewards["stay"], "stayed"
        elif action == "positive":
            # this APC is either True for positive/harmful or False for negative/benign
            if self.positive: # if the APC is positive
                return self.rewards["positive"]["TP"], "TP"
            else:
                return self.rewards["positive"]["FP"], "FP"
        elif action == "negative":
            if self.positive: # if the APC is positive
                return self.rewards["negative"]["FN"], "FN"
            else:
                return self.rewards["negative"]["TN"], "TN"
        raise NotImplementedError("How did we get here?")

    @staticmethod
    def eval_action_reward(action, reward):
        if action == "positive" and reward > 0: return "TP"
        elif action == "positive" and reward < 0: return "FP"
        elif action == "negative" and reward > 0: return "TN"
        elif action == "negative" and reward < 0: return "FN"
        else: return "unknown"

    def plotCertainty(self, tau=None, taus=None, title="Certainty over search time"):
        # taus must be a list of stopping points to plot
        if not isinstance(taus, list) and taus is not None: taus = [taus]
        elif taus is None and tau is not None and not isinstance(tau, list): taus = [tau]
        elif taus is None and tau is not None and isinstance(tau, list): taus = tau
        # ge the max time point to visualize
        max_t = 50 if taus is None else max(taus) * 1.2
        T = np.arange(max_t * 1.2)
        # visualize the certainty curve for many values of t
        env_pos = copy.copy(self)
        env_pos.positive = True
        # use copies of the env to not disturb any other workflow
        C1 = [env_pos.get_certainty(t) for t in T]
        # visualize the curve also for a version of APC with a inverted harmfulness
        env_neg = env_pos
        env_neg.positive = False
        # use copies of the env to not disturb any other workflow
        C2 = [env_neg.get_certainty(t) for t in T]
        # create the plot
        plt.figure(figsize=(7, 4))  # wide and flat
        plt.plot(T, C1, color="blue", label="positive APC")
        plt.plot(T, C2, color="red", label="negative APC")

        plt.axhline(y=env_pos.get_certainty(1), color='lightgray', linestyle='--') # certainty starting point
        if taus is not None:
            for tau in taus:
                plt.axvline(x=tau, color='green', linestyle='--') # stopping point tau

        plt.xlabel("Search time t")
        plt.ylabel("Certainty e(t)")
        plt.title(title)
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plotCertainties(biases:list, learning_rate=1, taus:list=None, title="Certainty over search time"):
        if not isinstance(taus, list) and taus is not None: taus = [taus]
        # create the plot
        plt.figure(figsize=(7, 4))  # wide and flat
        max_t = 50 if taus is None else max(taus) * 1.2
        T = np.arange(max_t * 1.2)
        for i, bias in enumerate(biases):
            # visualize the certainty curve for many values of t
            env_pos = StochasticAPC(bias=bias, learning_rate=learning_rate)
            env_pos.positive = True
            # use copies of the env to not disturb any other workflow
            C1 = [env_pos.get_certainty(t) for t in T]
            # visualize the curve also for a version of APC with a inverted harmfulness
            env_neg = env_pos
            env_neg.positive = False
            # use copies of the env to not disturb any other workflow
            C2 = [env_neg.get_certainty(t) for t in T]
            plt.plot(T, C1, color=plt.cm.tab10.colors[i], label=f"bias={bias}")
            plt.plot(T, C2, color=plt.cm.tab10.colors[i])

        plt.axhline(y=0.5, color='lightgray', linestyle='--') # certainty starting point
        if taus is not None:
            for tau in taus:
                plt.axvline(x=tau, color='green', linestyle='--') # stopping point tau

        plt.xlabel("Search time t")
        plt.ylabel("Certainty e(t)")
        plt.title(title)
        plt.ylim([0, 1])
        plt.grid(True)
        plt.legend()
        plt.show()
