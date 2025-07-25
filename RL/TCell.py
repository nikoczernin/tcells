
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
        Y = [self.env.get_certainty(t) for t in X]
        if self.env.isPositive:
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
    env = StochasticAPC()
    print(f"APC is _{'positive' if env.isPositive else 'negative'}_\n")
    agent = TCell_Threshold(env, threshold=.96)
    print(agent.policy)
    # agent.policy.plot(agent)
    R, t, transitions = agent.episode(verbose=True)
    final_action = transitions[-1][1]
    final_reward = transitions[-1][2]
    agent.plot_transitions(transitions)

    # print(f"Final action: {final_action}. Final reward: {final_reward}")
    # print("TCell decision evaluation:", env.eval_action_reward(final_action, final_reward))
    # test_TCell_Threshold()
