#!/usr/bin/env python3
"""Performs the backward algorithm for a hidden Markov model."""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Run the backward algorithm on an observation sequence.

    Args:
        Observation (numpy.ndarray): observation indices of shape (T,).
        Emission (numpy.ndarray): emission probabilities of shape (N, M).
        Transition (numpy.ndarray): transition probabilities of shape (N, N).
        Initial (numpy.ndarray): starting probabilities of shape (N, 1).

    Returns:
        (float, numpy.ndarray): the likelihood of the observations and the
        backward path probabilities of shape (N, T),
        or (None, None) on failure.
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    n, m = Emission.shape
    if not isinstance(Transition, np.ndarray) or \
            Transition.shape != (n, n):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (n, 1):
        return None, None
    if not _valid_probabilities(Emission) or \
            not _valid_probabilities(Transition) or \
            not _valid_probabilities(Initial):
        return None, None
    if not _valid_observations(Observation, m):
        return None, None
    if not np.isclose(np.sum(Initial), 1):
        return None, None

    Observation = Observation.astype(int)
    T = Observation.shape[0]
    B = np.zeros((n, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        for i in range(n):
            B[i, t] = np.sum(B[:, t + 1] * Transition[i] *
                             Emission[:, Observation[t + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
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
