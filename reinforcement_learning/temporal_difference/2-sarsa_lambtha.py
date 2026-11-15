#!/usr/bin/env python3
"""
Defines function that performs the SARSA(lambda) algorithm
"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    parameters:
        Q [numpy.ndarray of shape (s, a)]:
            contains the Q table
        state [int]:
            the current state
        epsilon [float]:
            the probability of exploring with a random action

    returns:
        the index of the next action
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(lambda) with accumulating eligibility traces

    parameters:
        env: the openAI environment instance
        Q [numpy.ndarray of shape (s, a)]:
            contains the Q table
        lambtha [float]:
            the eligibility trace factor
        episodes [int]:
            the total number of episodes to train over
        max_steps [int]:
            the maximum number of steps per episode
        alpha [float]:
            the learning rate
        gamma [float]:
            the discount rate
        epsilon [float]:
            the initial threshold for epsilon greedy
        min_epsilon [float]:
            the minimum value that epsilon should decay to
        epsilon_decay [float]:
            the decay rate for updating epsilon between episodes

    returns:
        Q: the updated Q table
    """
    initial_epsilon = epsilon
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        # how much credit each state-action pair gets for the current error
        traces = np.zeros_like(Q)
        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])
            # decay every trace, then reinforce the pair just taken
            traces *= gamma * lambtha
            traces[state, action] += 1
            Q += alpha * delta * traces
            if done:
                break
            state, action = next_state, next_action
        # decay epsilon exponentially towards min_epsilon
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * (episode + 1)))
    return Q
