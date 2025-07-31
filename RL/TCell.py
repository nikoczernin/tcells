
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from RL.APC import StochasticAPC
from RL.Environment import Environment
from RL.Policy import LinearPolicy, Policy, APCThresholdPolicy


class Agent():
    def __init__(self, env, T):
        self.T = T # max number of time steps the agent is allowed to take
        self.env = env
        self.policy = None

    def pick_action(self, state, epsilon):
        return self.policy.pick_action(state, epsilon)

    def episode(self, policy=None, epsilon=None, verbose=False):
        # performs one episode following the given policy and returns outcome
        # input: policy (dict), epsilon (float), verbose (bool); output: tuple (total reward, steps, transitions)
        # policy is an optional parameter, if none was given, use the Bot's own policy
        # if a policy was passed, use it as the behavioural policy (good for off-policy stuff)
        # 1 episode should look like this: {S0, A0, R1, S1, A1, R2, ..., ST-1, AT-1, RT}
        if policy is None: policy = self.policy
        s_t = self.env.starting_state
        transitions = []
        if verbose: print("Starting episode:")
        R, t = 0, 0
        for t in range(self.T):
            if verbose: self.env.print_state(s_t)
            # if S0 is already a terminal state, we still need to perform action A0 to get R1
            # just don't make an action -> a = None
            a = self.policy.pick_action(state=s_t, epsilon=epsilon)
            if verbose: print(f"a_{t}:", a)
            # move into a new state
            s_t_1 = self.env.apply_action(s_t, a)
            r = self.env.get_reward(s_t, a, s_t_1)
            R += r
            transitions.append((s_t, a, r, s_t_1))
            s_t = s_t_1
            if verbose: print()
            if self.env.state_is_terminal(s_t) or s_t is None:
                break
        if verbose:
            print("Finished episode at end-state:")
            self.env.print_state(s_t)
            print("Total reward:", R)
            print()
        return R, t, transitions


    def make_test_runs(self, k=100, *args, **kwargs):
        # performs multiple test episodes and prints average results
        # input: k (int), args/kwargs - additional params for episodes; output: none
        print(f"Performing {k} test runs ...")
        results = [self.episode(*args, **kwargs) for _ in range(k)]
        print("Best reward:", np.max([x[0] for x in results]))
        print("Mean reward:", np.mean([x[0] for x in results]))
        print("Mean time-step of termination:", np.mean([x[1] for x in results]))
        print()


    def plot_transitions(self, transitions):
        X = np.arange(1, len(transitions)+1)
        Y = [self.env.get_evidence(t) for t in X]
        if not self.env.positiveTendency:
            Y = [1-c for c in Y]
        plt.plot(X, Y)
        plt.xlabel("search time")
        plt.ylabel("q")
        plt.title("Perceived probability of APC being positive (q) over time")
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='gray', linestyle='dotted')
        plt.grid(True)
        plt.legend()
        plt.show()

class TCell(Agent):
    def __init__(self, env:Environment, T=100):
        # print("Initializing T-cell...")
        super().__init__(env, T=T)
        # set some policy
        pass


class TCell_Linear(TCell):
    def __init__(self, env:Environment, T=100):
        super().__init__(env, T)
        self.policy = LinearPolicy(env.actions, len(self.env.starting_state))


class TCell_Threshold(TCell):
    def __init__(self, env:Environment, T=100, threshold=.95):
        super().__init__(env, T)
        self.policy = APCThresholdPolicy(env.actions, threshold=threshold, T=T)







if __name__ == "__main__":
    # play a single episode with a TCell
    # use a TCell_Threshold
    # this TCell variant uses a simple policy that makes a decision solely based on q
    # if q is higher than the treshold or the final timestep is reached and q>0.5, it makes a positive classification
    # if q is lower than 1-threshold or the final timestep is reached and q<0.5, it makes a negative classification
    # otherwise it waits
    # if you don't change the threshold/policy, every episode will converge at the same time
    from RL.APC import StochasticAPC

    # StochasticAPC will pick a random value for isPositive, but you can also set it manually
    env = StochasticAPC()
    print(f"APC is _{'positive' if env.positiveTendency else 'negative'}_\n")
    agent = TCell_Threshold(env, threshold=.995)
    print(agent.policy)
    # agent.policy.plot(agent)
    R, t, transitions = agent.episode(verbose=True)
    final_action = transitions[-1][1]
    final_reward = transitions[-1][2]
    print(f"... and the env was {env.positiveTendency} ...")
    agent.plot_transitions(transitions)
