#!/usr/bin/env python3
"""
Defines function that performs the TD(lambda) algorithm
"""


import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm with accumulating eligibility traces

    parameters:
        env: the openAI environment instance
        V [numpy.ndarray of shape (s,)]:
            contains the value estimate
        policy [function]:
            takes in a state and returns the next action to take
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

    returns:
        V: the updated value estimate
    """
    for episode in range(episodes):
        state = env.reset()
        # how much credit each state gets for the current TD error
        traces = np.zeros_like(V)
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            delta = reward + gamma * V[next_state] - V[state]
            # decay every trace, then reinforce the state just visited
            traces *= gamma * lambtha
            traces[state] += 1
            V += alpha * delta * traces
            if done:
                break
            state = next_state
    return V
