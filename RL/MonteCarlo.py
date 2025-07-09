# Off- and On-Policy Monte Carlo Control for GridWorld
# Implements MC control for a 4×4 grid world using both on-policy and off-policy
# learning. Uses ε-greedy policies, returns tracking, and value updates to
# estimate the optimal policy π* via Monte Carlo sampling.

from pprint import pprint
import random
import numpy as np

# from GridWorld import GridWorld
from GridBot import Bot
from RL.APC import StochasticAPC


def MC_policy_control(bot: Bot, epsilon=.1, gamma=1, visit="first", off_policy=False, behaviour_policy=None, num_episodes=1000):
    # performs Monte Carlo policy control (on-policy or off-policy)
    # inputs: bot (Bot), epsilon (float), gamma (float), visit (str), off_policy (bool), behaviour_policy (dict), num_episodes (int)
    # outputs: none (updates bot.policy in place)
    if off_policy:
        behaviour_policy = bot.policy.copy()
    else: # on-policy: use the Bot's policy for the episodes, while also updating it
        behaviour_policy = bot.policy

    # estimate q with Q
    # Q is the action-value-function: maps from state to action to action-value
    Q = {}
    # dict: maps from state to action to a list of all returns received
    returns = {}
    for s in bot.env.state_generator():
        Q[s] = {a: 0.0 for a in bot.env.actions}
        returns[s] = {a: [] for a in bot.env.actions}

    for k in range(num_episodes):
        if k % (num_episodes//10) == 0:
            print(f"Iteration {k}")

        # generate an episode
        R_k, t_k, transitions = bot.episode(epsilon=epsilon, policy=behaviour_policy)
        # transitions looks like this: [(S0, A0, R1, S1), (S1, A1, R2, S2), ..., (ST-1, AT-1, RT, _)]
        g = 0
        # iterate over sequence backwards!
        for t in range(len(transitions)-1, -1, -1):
            # compute the return of the current time-step
            g = gamma * g + transitions[t][2]
            s_t = transitions[t][0]
            a_t = transitions[t][1]
            # for first visit MC, check if the state occurred before the current time-step, then save its return
            # for every visit MC, save its return anyway
            # TODO: moch des effizienter indem nicht jedes verfickte Mal alle previous_transitions gecheckt werden
            if any(previous_transition[0] == s_t for previous_transition in transitions[:t]) or visit == "every":
                # save g to the returns list for the current state and action
                returns[s_t][a_t].append(g)
                # update Q, the state-value mapping
                Q[s_t][a_t] = np.mean(returns[s_t][a_t])
                # get the action that leads to the maximum value in the current state according to Q
                # ties are broken randomly
                a_optimal = random.choice([
                    a for a, v in Q[s_t].items()
                    if v == max(Q[s_t].values())
                ])
                # update the Bot policy (not necessarily the behaviour_policy) for all actions in the current state
                for a in bot.env.actions:
                    # TODO: the following check is also in the Bot.policy class
                    # maybe we should generalize this ...
                    # if there is no policy rule for this state yet, create it
                    if s_t not in bot.policy.keys():
                        possible_actions = [a for a in bot.env.actions if bot.env.is_this_action_possible(s_t, a)]
                        bot.policy[s_t] = {a:1/len(possible_actions) for a in possible_actions}
                    bot.policy[s_t][a] = (1 - epsilon + epsilon/len(Q[s_t])) if a_optimal == a else epsilon/len(Q[s_t])
    print()
    # compute v damit ich es so wie der markus plotten kann amk
    v = {s: max(Q[s].values()) for s, a in Q.items()}
    return bot.policy, v


def test(env, epsilon=.4, num_episodes=1000, off_policy=True, verbose=False, num_test_runs=1000, T=100):
    # tests the environment by running episodes and policy optimization
    # inputs: env (Environment), epsilon (float), num_episodes (int), off_policy (bool), verbose (bool)
    # outputs: none (prints results)
    print("This is what the environment looks like:")
    print(env)
    bot = Bot(env=env, T = T, initialize_policy=False)
    # print("Policy before policy control:")
    # pprint(bot.policy)
    print("Now we run some episodes and see what we get (before optimizing the policy):")
    bot.make_test_runs(k=num_test_runs, verbose=verbose)
    print()
    print("##### Performing policy control #####")
    pi, v = MC_policy_control(bot, epsilon=epsilon, off_policy=off_policy, num_episodes=num_episodes)
    # print("Policy after policy control:")
    # bot.draw_policy()
    print("Full policy after policy control:")
    pprint(bot.policy)
    print()
    print()
    print("Value function after policy control:")
    bot.draw_v(v)
    print()
    print()
    print("Make more test runs after policy control:")
    bot.make_test_runs(k=num_test_runs)



# def test_grid_world(epsilon=0.1):
#     # tests Monte Carlo control on standard gridworld
#     h, w = 4, 4 # grid size
#     env = GridWorld(h, w, terminal_states=[(0, 0), (w-1, h-1)], starting_state=(2, 1))
#     test(env, epsilon)



def test_stochasticAPC(epsilon = .1):
    # tests Monte Carlo control on blackjack environment
    env = StochasticAPC()
    verbose = False
    bot = Bot(env=env, T = 20, initialize_policy=False)
    print("Policy before policy control:")
    pprint(bot.policy)
    print("Now we run some episodes and see what we get (before optimizing the policy):")
    bot.make_test_runs(k=50000, verbose=verbose)
    # print()
    # print("##### Performing policy control #####")
    # print()
    # MC_policy_control(bot, epsilon=epsilon, off_policy=True, num_episodes=1000000)
    # print("Policy after policy control:")
    # pprint(bot.policy)
    # bot.make_test_runs(k=5000)
    # print()
    # print("##### Performing a single episode with the new policy #####")
    # bot.episode(verbose=True)




def main():
    pass
    # test_grid_world()
    # test_windy_world()
    # test_tcell()
    # test_frozen_lake_hard()
    # test_frozen_lake()
    test_stochasticAPC()

if __name__ == '__main__':
    main()
