#!/usr/bin/env python3
"""
Defines function that uses epsilon-greedy to determine the next action
"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    parameters:
        Q [numpy.ndarray]:
            contains the Q-table
        state [int]:
            the current state
        epsilon [float]:
            the epsilon to use for the calculation

    returns:
        the next action index
    """
    p = np.random.uniform(0, 1)
    # exploit the best known action
    if p > epsilon:
        return np.argmax(Q[state])
    # explore with a random action
    return np.random.randint(0, Q.shape[1])
