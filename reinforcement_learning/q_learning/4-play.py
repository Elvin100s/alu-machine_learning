#!/usr/bin/env python3
"""
Defines function that has the trained agent play an episode
"""


import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode, displaying each state of the board

    parameters:
        env: the FrozenLakeEnv instance
        Q [numpy.ndarray]:
            contains the Q-table
        max_steps [int]:
            the maximum number of steps in the episode

    returns:
        the total rewards for the episode
    """
    state = env.reset()
    env.render()
    total_rewards = 0
    for step in range(max_steps):
        # always exploit the Q-table
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        env.render()
        total_rewards += reward
        if done:
            break
    return total_rewards
