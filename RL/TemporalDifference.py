from pprint import pprint
from random import random

import numpy as np
from matplotlib import pyplot as plt

from RL.APC import StochasticAPC
from RL.TCell import TCell
from utils import plot_blockwise_mean_rewards_line_graph, plot_line_graph


def SARSA(bot, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000, expected=False):
    if expected: print("Performing Expected-SARSA...")
    else: print("Performing SARSA...")
    # init action-value function (tabular, finite)
    # should we consider all states beforehand?
    Q = {s:{a:0 for a in bot.env.actions} for s in bot.env.state_generator()}
    total_rewards = [] # keep a list of total rewards for each episode
    for k in range(num_episodes):
        episode_reward = 0
        # reset the env
        bot.env.reset()
        # before every episode: set a policy according to Q
        for s in Q.keys():
            # update the policy of the bot
            bot.policy_set_action(s, max(Q[s], key=Q[s].get))

        # initialize s
        s_t = bot.env.starting_state
        # choose action a from s_t using e-greedy policy
        a_t = bot.pick_action(s_t, epsilon)
        for t in range(bot.T):
            # if s is terminal, terminate the episode
            if bot.env.state_is_terminal(s_t):
                break
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = bot.env.apply_action(s_t, a_t)
            r = bot.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # choose action a_t_1 from s_t_1 using e-greedy-policy
            a_t_1 = bot.pick_action(s_t_1, epsilon)
            # perform update of Q using SARSA update formula
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r - Q[s_t][a_t])
            else:
                if not expected:
                    Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r + gamma * Q[s_t_1][a_t_1] - Q[s_t][a_t])
                else:
                    Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r - Q[s_t][a_t] + sum([
                        bot.policy[s_t_1][a_t_1] * Q[s_t_1][a_t_1] for a_t_1 in bot.policy[s_t_1]
                    ]))

            # set s_t and a_t to the new state and action (we are doing on-policy control)
            s_t, a_t = s_t_1, a_t_1
        total_rewards.append(episode_reward)

    return Q, total_rewards

def expected_SARSA(bot, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000):
    return SARSA(bot, alpha, epsilon, gamma, num_episodes, expected=True)


def Q_Learning(agent, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000):
    print("Performing Q-Learning...")
    total_rewards = [] # keep a list of total rewards for each episode

    for k in range(num_episodes):
        if num_episodes < 20: print("Episode", k, "/", num_episodes)
        # reset the env
        agent.env.reset()
        episode_reward = 0
        # initialize s
        s_t = agent.env.starting_state
        for t in range(agent.T):
            if num_episodes < 20: print("Timestep", t, "/", agent.T)
            # print("State", s_t)
            # if s is terminal, terminate the episode
            if agent.env.state_is_terminal(s_t):
                break
            # choose action a from s_t using e-greedy policy
            a_t = agent.pick_action(s_t, epsilon)
            # print("Picked action:", a_t)
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = agent.env.apply_action(s_t, a_t)
            r = agent.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # choose action a_t_1 from s_t_1 that maximizes Q[s_t_1]
            # a_t_1 = max(Q[s_t], key=Q[s_t].get)
            # perform update of Q using SARSA update formula
            # if the next state is terminal, its future reward is zero
            # if bot.env.state_is_terminal(s_t_1):
            #     Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r - Q[s_t][a_t])
            # else:
            #     Q[s_t][a_t] = Q[s_t][a_t] + alpha * (r + gamma * Q[s_t_1][a_t_1] - Q[s_t][a_t])
            # update the policy
            if agent.env.state_is_terminal(s_t_1):
                target = r
            else:
                # get the greedy estimate of the next best action
                q_1 = [agent.policy.q(s_t_1, index_a) for index_a, a in enumerate(agent.env.actions)]
                # pick the action with the maximum action-value
                # discount it and add the reward, thats your action value for the previous state-action
                target = r + gamma * max(q_1)
            # compute the error = target - current estimate of the current state & action
            # print("Target reward:", target)
            index_a_t = agent.env.actions.index(a_t)
            error = target - agent.policy.q(s_t, index_a_t)
            # gradient step: ∂Q/∂W[a] = s
            agent.policy.W[index_a_t] += alpha * error * s_t
            # set s_t to the new state (off-policy control so a_t_1 does not get used)
            s_t = s_t_1
            # print()
        # print("Episode over. ")
        # print()
        total_rewards.append(episode_reward)
    print("Q-learning done")
    print()
    return total_rewards



def Double_Q_Learning(bot, alpha=.5, epsilon=.1, gamma=1.0, num_episodes=1000):
    print("Performing Double-Q-Learning...")
    total_rewards = [] # memory of all total episode-rewards
    # init 2 action-value functions (tabular, finite)
    Q1 = {s: {a: 0 for a in bot.env.actions} for s in bot.env.state_generator()}
    Q2 = Q1.copy()
    for k in range(num_episodes):
        # reset the env
        bot.env.reset()
        episode_reward = 0 # init the total episode reward
        # update the policy acc. to Q1 and Q2 (use avg action-values)
        for s in Q1.keys():
            # update the policy of the bot
            # get the action with the highest mean action-value
            avg_action_values = {a:(Q1[s][a] + Q2[s][a])/2 for a in Q1[s].keys()}
            bot.policy_set_action(s, max(avg_action_values, key=avg_action_values.get))

        # initialize s
        s_t = bot.env.starting_state
        for t in range(bot.T):
            # if s is terminal, terminate the episode
            if bot.env.state_is_terminal(s_t):
                break
            # choose action a from s_t using e-greedy policy
            a_t = bot.pick_action(s_t, epsilon)
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = bot.env.apply_action(s_t, a_t)
            r = bot.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # generate random uniform number, with a 50/50 chance swap them
            if random() < .5:
                Q1, Q2 = Q2, Q1
            # choose action a_t_1 from s_t_1 that maximizes Q[s_t_1]
            a_t_1 = max(Q1[s_t], key=Q1[s_t].get)
            # perform update of Q using SARSA update formula
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                Q1[s_t][a_t] = Q1[s_t][a_t] + alpha * (r - Q1[s_t][a_t])
            else:
                Q1[s_t][a_t] = Q1[s_t][a_t] + alpha * (r + gamma * Q2[s_t_1][a_t_1] - Q1[s_t][a_t])

            # set s_t to the new state (off-policy control so a_t_1 does not get used)
            s_t = s_t_1
        total_rewards.append(episode_reward)
    # compute an average Q
    Q = {s:{a:(Q1[s][a] + Q2[s][a])/2 for a in Q1[s].keys()} for s in Q1.keys()}

    return Q, total_rewards


def Speedy_Q_Learning(bot, alpha=None, gamma=1.0, epsilon=.1,  num_episodes=1000):
    print("Performing Speedy-Q-Learning...")
    # init action-value function (tabular, finite)
    Q = {s: {a: 0 for a in bot.env.actions} for s in bot.env.state_generator()}
    total_rewards = []
    for k in range(num_episodes):
        # reset the env
        bot.env.reset()
        episode_reward = 0
        for s in Q.keys():
            # update the policy of the bot
            bot.policy_set_action(s, max(Q[s], key=Q[s].get))

        # initialize s
        s_t = bot.env.starting_state
        for t in range(bot.T):
            # if s is terminal, terminate the episode
            if bot.env.state_is_terminal(s_t):
                break
            # set the learning_rate (dep on t)
            alpha_t = 1 / (t + 1)
            # choose action a from s_t using e-greedy policy
            a_t = bot.pick_action(s_t, epsilon)
            # take action a_t and receive s_t_1 and reward r
            s_t_1 = bot.env.apply_action(s_t, a_t)
            r = bot.env.get_reward(s_t, a_t, s_t_1)
            episode_reward += r
            # choose action a_t_1 from s_t_1 that maximizes Q[s_t_1]
            a_t_1 = max(Q[s_t], key=Q[s_t].get)
            # perform update of Q using SARSA update formula
            # before updating, save Q,r as Q_prev,r_prev
            Q_prev = Q.copy()
            r_prev = r
            # if the next state is terminal, its future reward is zero
            if bot.env.state_is_terminal(s_t_1):
                BQt = r
                BQt_1 = r_prev
                Q[s_t][a_t] = Q[s_t][a_t] + alpha_t * (BQt_1 - Q[s_t][a_t]) + (1 - alpha_t) * (BQt - BQt_1)

            else:
                BQt = r + gamma * Q[s_t_1][a_t_1]
                BQt_1 = r_prev + gamma * Q_prev[s_t_1][max(Q_prev[s_t_1], key=Q_prev[s_t_1].get)]
                Q[s_t][a_t] = Q[s_t][a_t] + alpha_t * (BQt_1 - Q[s_t][a_t]) + (1-alpha_t) * (BQt - BQt_1)

            # set s_t to the new state (off-policy control so a_t_1 does not get used)
            s_t = s_t_1
        total_rewards.append(episode_reward)
    print()
    return Q, total_rewards




def test_tcell(algo=Q_Learning, print_q=False, alpha = .2, epsilon = .3, gamma = .9, num_episodes=10000):
    print(f"Testing {algo.__name__} on TCell...")
    env = StochasticAPC()
    agent = TCell(env)
    alpha = .3
    epsilon = .2
    gamma = .6
    # num_episodes = 4000
    total_rewards = algo(agent, alpha=alpha, epsilon=epsilon, gamma=gamma, num_episodes=num_episodes)
    print(agent.policy)
    agent.plot_policy()
    # plot_blockwise_mean_rewards_line_graph(total_rewards, title=f"Total Rewards {algo.__name__}", xlabel="episodes", ylabel="reward")
    # bot.make_test_runs(1000)
    # print("-"*50)

