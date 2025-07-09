# RL environment for a single Stochastic APC
# a single APC has a certain probability (p) of containing a harmful antigen
# each timestep represents searching the APC more closely, which in turn costs time
# the longer an APC is investigated, the more certainty about whether it contains harmful antigen you should have

import random
from pprint import pprint

import numpy as np

from RL.Environment import Environment
from RL import utils

##### States, Actions & Rewards #####
# States:
# t: timestep. starts from 0, goes to infinity
# c: certainty. between 0 and 1. rises over time, converging to 1.
# stop: boolean. state is terminal
# different certainty functions may influence the behaviour

# Actions:
# stay: let the time advance and keep investigating the cell
# call: terminate and trigger an immune response (classify as positive)
# skip: terminate and trigger no immune response (classify as negative)

# Rewards:
# stay: -1
# call & pos (TP): 100
# call & neg (FP): -100
# skip & pos (FN): -100
# skip & neg (TN): 100


class StochasticAPC(Environment):
    def __init__(self, certainty_fun=utils.rational_function):
        # actions
        actions = ["stay", "call", "skip"]
        super().__init__(actions)
        # rewards
        self.rewards = {
            "stay": -1,
            "call": {
                "TP": 100,
                "FP": -100,
            },
            "skip": {
                "TN": 100,
                "FN": -100,
            }
        }
        # starting_state
        self.starting_state = np.array([0., 0., 0.])
        self.certainty_fun = certainty_fun


    def state_is_terminal(self, state) -> bool:
        return state[2] == 1

    def print_state(self, state):
        print(f"State t={int(state[0])}, c={round(state[1], 4)}, stop={bool(state[2])}")

    def get_certainty(self, t):
        # apply a certainty function
        # you can put any other function here
        # it should rise monotonously and ideally converge to 1
        return self.certainty_fun(t)

    def apply_action(self, state, action):
        new_state = state.copy()
        # advance time
        new_state[0] += 1
        # recompute certainty
        new_state[1] = self.get_certainty(new_state[0])
        # if actions are skip or call, terminate
        if action == "call" or action == "skip":
            new_state[2] = 1
        return new_state

    def get_reward(self, state: tuple, action: tuple, new_state: tuple = None):
        # if the agent wants to inspect the APC for longer
        # his action was "stay"
        # penalize him for the time it takes to investigate
        if action == "stay":
            return self.rewards["stay"]
        # whatever the actual harmfulness of the APC
        # whether the agent is correct is
        # only dependent on the certainty
        certainty = state[1]
        guess_is_correct = random.random() < certainty
        # return for the picked action the appropriate reward
        if action == "call":
            if guess_is_correct: return self.rewards["call"]["TP"]
            else: return self.rewards["call"]["FP"]
        elif action == "skip":
            if guess_is_correct: return self.rewards["skip"]["TN"]
            else: return self.rewards["skip"]["FN"]

    def eval_action_reward(self, action, reward):
        if action == "call" and reward > 0: return "TP"
        elif action == "call" and reward < 0: return "FP"
        elif action == "skip" and reward > 0: return "TN"
        elif action == "skip" and reward < 0: return "FN"
        else: return "unknown"
