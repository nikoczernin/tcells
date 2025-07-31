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
# different certainty functions may influence the behaviour

# Hidden State:
# positiveTendency = True/False
# each APC has a tendency of being harmful, which is either positive (True) or negative (False)
# this is hiddne to an agent and only affects the direction of the evidence function e
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


class StochasticAPC(Environment):
    def __init__(self, certainty_fun=utils.rational_function):
        # actions
        actions = ["stay", "positive", "negative"]
        super().__init__(actions)
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
        self.certainty_fun = certainty_fun
        self.isLikelyPositive = None
        self.reset()

    def reset(self):
        self.positiveTendency = np.random.choice([True, False])

    def state_is_terminal(self, state) -> bool:
        return state[2] == 1

    @staticmethod
    def print_state(state):
        print(f"State t={int(state[0])}, q={round(state[1], 4)}, stop={bool(state[2])}")

    def get_evidence(self, t):
        # returns a value between 0.5 and 1
        # the passed function must rise monotonously and converge to 1
        # at t=0, it is 0
        # at t=1, it is 0.5
        # at t -> inf, it converges to 1
        if t == 0: return .5
        return self.certainty_fun(t)

    def apply_action(self, state, action):
        new_state = state.copy()
        # advance t (time)
        new_state[0] += 1
        # recompute q (perceived probability of APC being positive)
        # it is the certainty if the APC is positive, and thus converges from 0.5 to 1 at t increases,
        # or it is 1-certainty if the APC is negative, converging from 0.5 to 0 at t increases
        certainty = self.get_evidence(new_state[0])
        new_state[1] = certainty if self.positiveTendency else 1 - certainty
        # if actions are negative or positive, terminate
        if action == "positive" or action == "negative":
            new_state[2] = 1
        return new_state

    def get_reward(self, state: tuple, action: tuple, new_state: tuple = None):
        # if the agent wants to inspect the APC for longer
        # his action was "stay"
        # penalize him for the time it takes to investigate
        if action == "stay":
            return self.rewards["stay"]
        # generate the status of this APC
        # it is either True for positive/harmful or False for negative/benign
        # the status is determined by sampling from the uniform distribution
        # the probability is affected by the amount of evidence collected
        # evidence is collected by advancing the e function
        e_t = state[1]
        isPositive = random.random() < e_t
        # return for the picked action the appropriate reward
        if isPositive:
            if action == "positive":
                return self.rewards["positive"]["TP"]
            elif action == "negative":
                return self.rewards["negative"]["FN"]
        else:
            if action == "positive":
                return self.rewards["positive"]["FP"]
            elif action == "negative":
                return self.rewards["negative"]["TN"]

    @staticmethod
    def eval_action_reward(action, reward):
        if action == "positive" and reward > 0: return "TP"
        elif action == "positive" and reward < 0: return "FP"
        elif action == "negative" and reward > 0: return "TN"
        elif action == "negative" and reward < 0: return "FN"
        else: return "unknown"
