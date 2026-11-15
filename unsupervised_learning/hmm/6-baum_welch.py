#!/usr/bin/env python3
"""
Defines function that performs the Baum-Welch algorithm for Hidden Markov Model
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a Hidden Markov Model

    parameters:
        Observation [numpy.ndarray of shape (T,)]:
            contains the index of the observation
        Emission [numpy.ndarray of shape (M, N)]:
            contains the emission probabilities
        Transition [numpy.ndarray of shape (M, M)]:
            contains the transition probabilities
        Initial [numpy.ndarray of shape (M, 1)]:
            contains the starting probabilities

    returns:
        P, F:
            P [float]: the likelihood of the observations given the model
            F [numpy.ndarray of shape (M, T)]:
                contains the forward path probabilities
    """
    T = Observation.shape[0]
    M = Emission.shape[0]
    F = np.zeros((M, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = np.sum(F[:, t - 1].reshape(-1, 1) * Transition *
                         Emission[:, Observation[t]], axis=0)
    P = np.sum(F[:, T - 1])
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a Hidden Markov Model

    parameters:
        Observation [numpy.ndarray of shape (T,)]:
            contains the index of the observation
        Emission [numpy.ndarray of shape (M, N)]:
            contains the emission probabilities
        Transition [numpy.ndarray of shape (M, M)]:
            contains the transition probabilities
        Initial [numpy.ndarray of shape (M, 1)]:
            contains the starting probabilities

    returns:
        P, B:
            P [float]: the likelihood of the observations given the model
            B [numpy.ndarray of shape (M, T)]:
                contains the backward path probabilities
    """
    T = Observation.shape[0]
    M = Emission.shape[0]
    B = np.zeros((M, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(B[:, t + 1] * Transition *
                         Emission[:, Observation[t + 1]], axis=1)
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a Hidden Markov Model

    parameters:
        Observations [numpy.ndarray of shape (T,)]:
            contains the index of the observation
            T: number of observations
        Transition [2D numpy.ndarray of shape (M, M)]:
            contains the initialized transition probabilities
            M: the number of hidden states
        Emission [numpy.ndarray of shape (M, N)]:
            contains the initialized emission probabilities
            N: number of output states
        Initial [numpy.ndarray of shape (M, 1)]:
            contains the initialized starting probabilities
        iterations [positive int]:
            the number of times expectation-maximization should be performed

    returns:
        the converged Transition, Emission
        or None, None on failure
    """
    # check that Observations is the correct type and dimension
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    # save T from Observations' shape
    T = Observations.shape[0]
    # check that Transition is the correct type and dimension
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    # save M and check that Transition is square
    M, M_check = Transition.shape
    if M != M_check:
        return None, None
    # check that Emission is the correct type and dimension
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    # check that Emission's dimension matches M from Transition and save N
    M_check, N = Emission.shape
    if M_check != M:
        return None, None
    # check that Initial is the correct type and dimension
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    # check that Initial's dimensions match (M, 1)
    M_check, one = Initial.shape
    if M_check != M or one != 1:
        return None, None
    # check that iterations is a positive int
    if type(iterations) is not int or iterations < 1:
        return None, None
    # work on copies so the caller's arrays are not modified
    Transition = Transition.copy()
    Emission = Emission.copy()
    # one_hot[k, t] is 1 when observation k was seen at time t
    one_hot = (Observations == np.arange(N).reshape(-1, 1)).astype(float)
    for _ in range(iterations):
        # expectation step
        P, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        if P == 0:
            return None, None
        # xi[i, j, t]: probability of being in i at t and in j at t + 1
        xi = (alpha[:, np.newaxis, :T - 1] *
              Transition[:, :, np.newaxis] *
              (Emission[:, Observations[1:]] * beta[:, 1:])[np.newaxis]) / P
        # gamma[i, t]: probability of being in i at t
        gamma = alpha * beta / P
        # maximization step
        Transition = (np.sum(xi, axis=2) /
                      np.sum(gamma[:, :T - 1], axis=1).reshape(-1, 1))
        Emission = (np.matmul(gamma, one_hot.T) /
                    np.sum(gamma, axis=1).reshape(-1, 1))
    return Transition, Emission
