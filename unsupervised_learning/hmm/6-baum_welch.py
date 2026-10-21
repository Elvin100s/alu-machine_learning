#!/usr/bin/env python3
"""Trains a hidden Markov model with the Baum-Welch algorithm."""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Run the Baum-Welch expectation-maximization algorithm.

    Args:
        Observations (numpy.ndarray): observation indices of shape (T,).
        Transition (numpy.ndarray): initialized transitions of shape (M, M).
        Emission (numpy.ndarray): initialized emissions of shape (M, N).
        Initial (numpy.ndarray): initialized starting probabilities (M, 1).
        iterations (int): number of expectation-maximization iterations.

    Returns:
        (numpy.ndarray, numpy.ndarray): the converged transition and emission
        probabilities, or (None, None) on failure.
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    states, symbols = Emission.shape
    if not isinstance(Transition, np.ndarray) or \
            Transition.shape != (states, states):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (states, 1):
        return None, None
    if not _valid_probabilities(Emission) or \
            not _valid_probabilities(Transition) or \
            not _valid_probabilities(Initial):
        return None, None
    if not _valid_observations(Observations, symbols):
        return None, None
    if not np.isclose(np.sum(Initial), 1):
        return None, None
    if Observations.shape[0] < 2:
        return None, None
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        return None, None

    Observations = Observations.astype(int)
    T = Observations.shape[0]
    Transition = Transition.copy()
    Emission = Emission.copy()
    for _ in range(iterations):
        _, alpha = _forward(Observations, Emission, Transition, Initial)
        _, beta = _backward(Observations, Emission, Transition, Initial)

        # xi[i, j, t]: probability of being in state i at t and state j at t+1
        emitted = (Emission[:, Observations[1:]] * beta[:, 1:]).T
        xi = (alpha[:, :-1][:, None, :] * Transition[:, :, None] *
              emitted.T[None, :, :])
        total = np.sum(xi, axis=(0, 1))
        if not np.all(total > 0):
            return None, None
        xi = xi / total

        # gamma[i, t]: probability of being in state i at time t
        gamma = np.zeros((states, T))
        gamma[:, :T - 1] = np.sum(xi, axis=1)
        gamma[:, T - 1] = np.sum(xi[:, :, T - 2], axis=0)

        # maximization step
        transition_den = np.sum(gamma[:, :T - 1], axis=1)
        transition_num = np.sum(xi, axis=2)
        for i in range(states):
            if transition_den[i] > 0:
                Transition[i] = transition_num[i] / transition_den[i]

        emission_den = np.sum(gamma, axis=1)
        for k in range(symbols):
            emission_num = np.sum(gamma[:, Observations == k], axis=1)
            for i in range(states):
                if emission_den[i] > 0:
                    Emission[i, k] = emission_num[i] / emission_den[i]
    return Transition, Emission


def _forward(Observation, Emission, Transition, Initial):
    """Compute the forward path probabilities.

    Args:
        Observation (numpy.ndarray): observation indices of shape (T,).
        Emission (numpy.ndarray): emission probabilities of shape (N, M).
        Transition (numpy.ndarray): transition probabilities of shape (N, N).
        Initial (numpy.ndarray): starting probabilities of shape (N, 1).

    Returns:
        (float, numpy.ndarray): likelihood and forward probabilities (N, T).
    """
    T = Observation.shape[0]
    F = np.zeros((Emission.shape[0], T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = (F[:, t - 1] @ Transition) * Emission[:, Observation[t]]
    return np.sum(F[:, T - 1]), F


def _backward(Observation, Emission, Transition, Initial):
    """Compute the backward path probabilities.

    Args:
        Observation (numpy.ndarray): observation indices of shape (T,).
        Emission (numpy.ndarray): emission probabilities of shape (N, M).
        Transition (numpy.ndarray): transition probabilities of shape (N, N).
        Initial (numpy.ndarray): starting probabilities of shape (N, 1).

    Returns:
        (float, numpy.ndarray): likelihood and backward probabilities (N, T).
    """
    states, _ = Emission.shape
    T = Observation.shape[0]
    B = np.zeros((states, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        B[:, t] = Transition @ (Emission[:, Observation[t + 1]] *
                                B[:, t + 1])
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def _valid_probabilities(matrix):
    """Check that a matrix holds finite probabilities between 0 and 1.

    Args:
        matrix (numpy.ndarray): values to check.

    Returns:
        bool: True if every value is a valid probability, False otherwise.
    """
    if not np.issubdtype(matrix.dtype, np.number) or \
            np.iscomplexobj(matrix):
        return False
    return bool(np.all(np.isfinite(matrix)) and
                not (matrix < 0).any() and not (matrix > 1).any())


def _valid_observations(Observation, m):
    """Check that observations are integer indices inside the emission range.

    Args:
        Observation (numpy.ndarray): observation indices.
        m (int): number of possible observations.

    Returns:
        bool: True if every observation index is valid, False otherwise.
    """
    if Observation.shape[0] == 0:
        return False
    if not np.issubdtype(Observation.dtype, np.number) or \
            np.iscomplexobj(Observation):
        return False
    if not np.issubdtype(Observation.dtype, np.integer):
        if not np.all(np.equal(np.mod(Observation, 1), 0)):
            return False
    return bool((Observation >= 0).all() and (Observation < m).all())
