# Implements a reinforcement learning agent (Bot) that interacts
# with a GridWorld environment. Includes policy evaluation,
# policy iteration, and value iteration algorithms.



import random

import numpy as np


class Bot():
    def __init__(self, env, T=100, initialize_policy=True):
        self.T = T # max number of time steps the agent is allowed to take
        # the environment contains states and actions
        self.env = env
        # in the scriptum the policy is often referred to as a vector of length S, containing actions
        # I opted to use a dictionary that maps from states to actions to probabilities of using that action in that state
        # self.policy: pi(a|s) -> [0, 1]
        # I initialize them as equiprobable, which leads to random action-picking

        # initialize_policy: boolean
        # you may not want to initialize a policy, because e.g. in MC, precomputing all states may
        # not be necessary
        self.policy = {}
        if initialize_policy:
            self.init_policy()

    def init_policy(self, hardline=False):
        # initializes the policy with equal probabilities for possible actions
        # input: hardline (bool); output: none
        for s in self.env.state_generator():
            possible_actions = [a for a in self.env.actions if (self.env.is_this_action_possible(s, a))]
            self.policy[s] = {}
            for a in self.env.actions:
                self.policy[s].update({a: 1 / len(possible_actions) if a in possible_actions else 0})
        if hardline: self.hardline_policy()

    # policy is typically a probability of all possible actions in a given state
    # when an action is performed, the policy returns each action with a certain probability
    # if you want to max out the probability of the most probable actions to get only probs of either 0 or 1
    # then this function is for you
    def hardline_policy(self):
        # makes the policy deterministic (probabilities either 0 or 1)
        # input/output: none
        for s in self.policy.keys():
            # get a random pick of the most probable actions
            a_max = self.pick_action(s)
            # now set all probabilities to zero, except for the max one
            for a in self.policy[s].keys():
                self.policy[s][a] = 1 if a == a_max else 0

    def policy_set_action(self, s, a_new, policy = None):
        # with this function you can update a policy to set all actions for a given state to a probability of 0
        # except for the action a_new, which will get a probability of 1
        if policy is None: policy = self.policy
        for a in policy[s].keys():
            policy[s][a] = 1 if a == a_new else 0


    # function for picking the best action from a policy, draws are resolved randomly
    # you can supply an epsilon for epsilon greedy exploration-exploitation
    def pick_action(self, s_t, epsilon=-1, policy=None):
        # updates the policy to set probability=1 for a specific action
        # input: s (state), a_new (action), policy (dict or None); output: none
        if policy is None:
            policy = self.policy
        if self.env.state_is_terminal(s_t):
            return None
        # TODO: if the state is unknown to the policy, compute a new ruleset
        if s_t not in policy.keys():
            possible_actions = [a for a in self.env.actions if self.env.is_this_action_possible(s_t, a)]
            policy[s_t] = {a:1/len(possible_actions) for a in possible_actions}

        # generate a random uniform number
        # if it is smaller than epsilon, we explore, otherwise we exploit normally
        ##### EXPLORE #####
        if random.random() < epsilon:
            # do not pick any impossible action though
            return random.choice(list(a for a in policy[s_t].keys() if self.env.is_this_action_possible(s_t, a)))
        ##### EXPLOIT #####
        else:
            # get the action with the highest probability given the current state from the policy
            max_prob = max(policy[s_t].values())
            # get all actions for this state that have the max probability
            best_keys = [k for k, v in policy[s_t].items() if v == max_prob]
            # pick a random action from the most likely ones
            chosen_key = random.choice(best_keys)
            return chosen_key


    def value_fun(self, s_t, t, gamma):
        # computes the value function recursively based on the current policy
        # input: s_t (state), t (int), gamma (float); output: float
        total_reward = 0
        # if this recursion is either at a terminal state or at maximum depth: return without any reward
        if s_t in self.env.state_is_terminal(s_t) or t >= self.T:
            return total_reward
        # for every action that is possible from this state
        for a in self.policy[s_t].keys():
            # get the probability of performing the action
            action_prob = self.policy[s_t][a]
            if action_prob: # > 0, otherwise no need to go deeper in the search tree here
                action_reward = self.action_value_fun(s_t, a, t, gamma)
                total_reward += action_reward * action_prob
        # return the sum of the expected reward in this state given the policy
        return total_reward


    def action_value_fun(self, s_t, a, t, gamma):
        # computes action-value function recursively using transitions
        # input: s_t (state), a (action), t (int), gamma (float); output: float
        state_transition_probs = self.env.get_possible_outcomes(s_t, a)
        # the future possible rewards are computed recursively using the value function, that means
        # any future decisions are made with the policy rather than a fixed function
        total = 0
        for s_t_1, prob in state_transition_probs.items():
            # Get the immediate reward for transitioning from s_t to s_t_1 using action a
            reward = self.env.get_reward(s_t, a, s_t_1)
            # Recursively compute the estimated future value
            future_value = gamma * self.value_fun(s_t_1, t + 1, gamma)
            # Add the weighted value (probability × (reward + future value))
            total += prob * (reward + future_value)
        return total

    def action_value_fun_star(self, s, a, gamma, v):
        # computes optimal action-value given current value estimates v
        # input: s (state), a (action), gamma (float), v (dict); output: float
        state_transition_probs = self.env.get_possible_outcomes(s, a)
        # if s_t_1 is a terminal state, its reward will be zero because were already at the goal
        # otherwise just return the optimal next action's expected reward discounted by gamma
        total = 0
        for s_t_1, prob in state_transition_probs.items():
            # Get immediate reward
            reward = self.env.get_reward(s, a, s_t_1)
            # If the next state is not terminal, add discounted future value
            if not self.env.state_is_terminal(s_t_1):
                try:
                    future_value = gamma * v[s_t_1]
                except KeyError:
                    future_value = 0
            else:
                future_value = 0
            # Weighted contribution of this transition
            total += prob * (reward + future_value)
        return total

    def draw_v(self, v):
        # draws the grid with the current value function v
        # input: v (dict); output: none
        self.env.grid.draw_grid(v, round_to=2)
        print()

    def draw_policy(self):
        # visualizes the current policy on the grid
        print("Current policy")
        pi = {s:self.env.action_str(self.pick_action(s)) for s in self.policy.keys()}
        if isinstance(self.env, GridWorld):
            self.env.grid.draw_grid(pi)
            print()

    def episode(self, policy=None, epsilon=-1, verbose=False):
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
            if verbose: print(f"t_{t}: {s_t}")
            # if S0 is already a terminal state, we still need to perform action A0 to get R1
            # just don't make an action -> a = None
            a = self.pick_action(s_t, epsilon=epsilon, policy=policy)
            if verbose: print(f"a_{t}:", a)
            # move into a new state
            outcomes_dict = self.env.get_possible_outcomes(s_t, a)
            # the transition includes
            s_t_1 = self.env.resolve_outcome(outcomes_dict)
            r = self.env.get_reward(s_t, a, s_t_1)
            R += r
            transitions.append((s_t, a, r, s_t_1))
            s_t = s_t_1
            # if verbose: print()
            if self.env.state_is_terminal(s_t) or s_t is None:
                break
        if verbose:
            print("Finished episode at end-state:")
            print(s_t)
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

if __name__ == "__main__":
    env = BlackJack()
    env.set_start()
    bot = Bot(env, T=20)
    bot.episode(verbose=True)
    bot.episode(verbose=True)
