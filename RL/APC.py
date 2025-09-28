# RL environment for a single Stochastic APC
# each timestep represents searching the APC more closely, which in turn costs time
# the longer an APC is investigated, the more certainty about whether it contains harmful antigen you should have

import random
from pprint import pprint

import numpy as np

from RL.Environment import Environment
from RL import utils

##### States, Actions & Rewards #####
# States: Each state is an array of three values:
# 1. the timestep `t`, natural numbers
# 2. the evidence function = recorded probability of the APC being positive `e` $\in [0, 1]$.
    # This starts at `0.5` and gets nudged in either direction toward 0 or 1, depending on the attribute `isPositive`.
# 3. `1` if this state is a terminal state, `0` at the start of an episode and made positive by the agent

# Hidden State:
# positive = True/False
# each APC has a probability of being harmful/positive (True) or benign/negative (False)
# this is hidden to an agent
# this gets assigned randomly at environment-creation or -reset

# Actions:
# - `stay`: advance to the next timestep without making a decision
# - `classify`: make a classification. it will be `positive` with probability c(t) or `negative` with prob 1-c(t)
    # - `positive`: classify the APC to be `positive`
    # - `negative`: classify the APC to be `negative`


# Rewards:
# stay: -1
# positive & pos (TP): 100
# positive & neg (FP): -100
# negative & pos (FN): -100
# negative & neg (TN): 100


class StochasticAPC(Environment):
    # ACTIONS
    ACTIONS = ["stay", "classify"]
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
                 learning_rate:float=1.0):
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
        # starting_state
        self.starting_state = np.array([0., 0.5, 0.])
        # probability of APC being positive
        # sets the starting point of the certainty function
        self.bias = bias / (1 - bias + (1e-12 if bias == 1 else 0))
        self.learning_rate = learning_rate * 0.5
        self.certainty_fun = certainty_fun
        # set the status
        self.p = p
        self.positive = None
        self.reset()

    def reset(self):
        self.positive = random.random() < self.p

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
            certainty = self.certainty_fun(t * self.bias, flatness=self.learning_rate)
        else:
            # adjust t by the inverted bias
            certainty = 1 -self.certainty_fun(t / self.bias, flatness=self.learning_rate)
        return certainty

    def apply_action(self, state, action):
        new_state = state.copy()
        # advance t (time)
        new_state[0] += 1
        # recompute certainty e (certainty of APC being positive)
        certainty = self.get_certainty(new_state[0])
        # print("Certainty=", certainty)
        new_state[1] = certainty
        # if ACTIONS are negative or positive, terminate
        if action == "classify":
            new_state[2] = 1
        return new_state

    def get_reward(self, state: tuple, action: tuple, new_state: tuple = None):
        # state, action, new_state -> reward, event
        # if the agent wants to inspect the APC for longer
        # his action was "stay"
        # penalize him for the time it takes to investigate
        if action == "stay":
            return self.rewards["stay"], "stayed"
        elif action == "classify":
            # this APC is either True for positive/harmful or False for negative/benign
            # the ImmuneSystem will make a positive classification with probability c(t)
            # or a negative classification with probability 1-c(t)
            # the decision is determined by sampling from the uniform distribution
            certainty = state[1]
            positive_classification = random.random() < certainty
            # print("Classifying", "positive" if positive_classification else "negative")
            # return for the picked action the appropriate reward
            if self.positive: # if the APC is positive
                if positive_classification:
                    return self.rewards["positive"]["TP"], "TP"
                else:
                    return self.rewards["negative"]["FN"], "FN"
            else:
                if positive_classification:
                    return self.rewards["positive"]["FP"], "FP"
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
