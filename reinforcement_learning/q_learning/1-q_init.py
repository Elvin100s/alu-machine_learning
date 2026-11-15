#!/usr/bin/env python3
"""
Defines function that initializes the Q-table
"""


import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    parameters:
        env: the FrozenLakeEnv instance

    returns:
        the Q-table as a numpy.ndarray of zeros of shape (states, actions)
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return Q
