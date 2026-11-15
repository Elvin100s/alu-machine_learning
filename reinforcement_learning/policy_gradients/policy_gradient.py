#!/usr/bin/env python3
"""
Defines functions that compute a softmax policy and its
Monte-Carlo policy gradient
"""


import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    parameters:
        matrix [numpy.ndarray of shape (1, s)]:
            the current observation of the environment
        weight [numpy.ndarray of shape (s, a)]:
            the weight matrix

    returns:
        [numpy.ndarray of shape (1, a)]:
            the softmax probability of taking each action
    """
    z = np.exp(np.matmul(matrix, weight))
    return z / np.sum(z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a weight

    parameters:
        state [numpy.ndarray of shape (1, s)]:
            the current observation of the environment
        weight [numpy.ndarray of shape (s, a)]:
            the weight matrix

    returns:
        action, gradient:
            action [int]: the action sampled from the policy
            gradient [numpy.ndarray of shape (s, a)]:
                the gradient of log(policy[action]) with respect to weight
    """
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])
    # d log(softmax[action]) / dz = one_hot(action) - softmax
    dlog = -probs
    dlog[0, action] += 1
    gradient = np.matmul(state.T, dlog)
    return action, gradient
