
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from RL.APC import StochasticAPC
from RL.Environment import Environment
from RL.Policy import LinearPolicy, Policy, SingleSearchPhasePolicy, DoubleSearchPhasePolicy


class Agent():
    def __init__(self, env, T, verbose=False):
        self.T = T # max number of time steps the agent is allowed to take
        self.env = env
        self.policy = None
        self.verbose = verbose

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
            r, eval = self.env.get_reward(s_t, a, s_t_1)
            R += r
            transitions.append((s_t, a, r, s_t_1, eval))
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
        # if not self.env.positive:
        #     Y = [1-c for c in Y]
        plt.plot(X, Y)
        plt.xlabel("search time")
        plt.ylabel("certainty c(t)")
        plt.title("Certainty of APC being positive over time")
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='gray', linestyle='dotted')
        plt.grid(True)
        plt.legend()
        plt.show()


class ImmuneSystem(Agent):
    def __init__(self, env:StochasticAPC, T, verbose=False):
        # print("Initializing T-cell...")
        super().__init__(env, T, verbose=verbose)
        # set some policy
        pass


# class ImmuneSystem_Linear(ImmuneSystem):
#     def __init__(self, env:StochasticAPC, T=100):
#         super().__init__(env, T)
#         self.policy = LinearPolicy(env.ACTIONS, len(self.env.starting_state))


class ImmuneSystem_SinglePhase(ImmuneSystem):
    def __init__(self, env:StochasticAPC, tau, verbose=False):
        super().__init__(env, tau*10, verbose=verbose) # set a super high max time to be safe
        self.tau = tau
        self.policy = SingleSearchPhasePolicy(env.ACTIONS, tau=tau)

    def str(self):
        return f"ImmuneSystem_SinglePhase [tau={self.tau}]"


class ImmuneSystem_DoublePhase(ImmuneSystem):
    def __init__(self, env:StochasticAPC, tau_1, tau_2, verbose=False):
        super().__init__(env, tau_2*10, verbose=verbose)
        self.tau_1, self.tau_2 = tau_1, tau_2
        self.policy = DoubleSearchPhasePolicy(env.ACTIONS, tau_1=tau_1, tau_2=tau_2)

    def str(self):
        return f"ImmuneSystem_SinglePhase [tau={self.tau_1}, {self.tau_2}]"




def test_single():
    # play a single episode with a ImmuneSystem
    from RL.APC import StochasticAPC
    # StochasticAPC will pick a random value for isPositive, but you can also set it manually
    env = StochasticAPC(learning_rate=.8, p=0.5, bias=.9995)
    tau = 10
    agent = ImmuneSystem_SinglePhase(env, tau=tau, verbose=True)
    env.plotCertainty(tau=tau)
    R, t, transitions = agent.episode()
    print()
    print(f"APC is _{'positive' if env.positive else 'negative'}_")
    print("Time taken:", t)
    final_action = transitions[-1][1]
    final_reward = transitions[-1][2]
    eval = transitions[-1][4]
    print(eval, "->", R)


def test_double():
    # play a single episode with a ImmuneSystem
    from RL.APC import StochasticAPC
    # StochasticAPC will pick a random value for isPositive, but you can also set it manually
    env = StochasticAPC(learning_rate=2, p=0.1, bias=.9)
    tau_1, tau_2 = 3, 6
    agent = ImmuneSystem_DoublePhase(env, tau_1=tau_1, tau_2= tau_2, verbose=True)
    env.plotCertainty(taus=[tau_1, tau_2])
    R, t, transitions = agent.episode()
    print()
    print(f"APC is _{'positive' if env.positive else 'negative'}_")
    print("Time taken:", t)
    final_action = transitions[-1][1]
    final_reward = transitions[-1][2]
    eval = transitions[-1][4]
    print(eval, "->", R)



if __name__ == "__main__":
    # test_single()
    test_double()



