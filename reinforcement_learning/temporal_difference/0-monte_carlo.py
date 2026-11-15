#!/usr/bin/env python3
"""
Defines function that performs the Monte Carlo algorithm
"""


import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm

    parameters:
        env: the openAI environment instance
        V [numpy.ndarray of shape (s,)]:
            contains the value estimate
        policy [function]:
            takes in a state and returns the next action to take
        episodes [int]:
            the total number of episodes to train over
        max_steps [int]:
            the maximum number of steps per episode
        alpha [float]:
            the learning rate
        gamma [float]:
            the discount rate

    returns:
        V: the updated value estimate
    """
    for episode in range(episodes):
        state = env.reset()
        history = []
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            history.append([state, reward])
            if done:
                break
            state = next_state
        history = np.array(history, dtype=int)
        G = 0
        # walk the episode backwards, accumulating the discounted return
        for state, reward in history[::-1]:
            G = gamma * G + reward
            # only update states that are not among the first `episode` steps
            if state not in history[:episode, 0]:
                V[state] += alpha * (G - V[state])
    return V
