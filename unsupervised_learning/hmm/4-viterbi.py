#!/usr/bin/env python3
"""Finds the most likely hidden state sequence with the Viterbi algorithm."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculate the most likely sequence of hidden states for an HMM.

    Args:
        Observation (numpy.ndarray): observation indices of shape (T,).
        Emission (numpy.ndarray): emission probabilities of shape (N, M).
        Transition (numpy.ndarray): transition probabilities of shape (N, N).
        Initial (numpy.ndarray): starting probabilities of shape (N, 1).

    Returns:
        (list, float): the most likely sequence of hidden states of length T
        and the probability of obtaining that sequence,
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
    delta = np.zeros((n, T))
    backpointer = np.zeros((n, T - 1), dtype=int)
    delta[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        scores = delta[:, t - 1][:, None] * Transition
        backpointer[:, t - 1] = np.argmax(scores, axis=0)
        delta[:, t] = np.max(scores, axis=0) * Emission[:, Observation[t]]

    path = [int(np.argmax(delta[:, T - 1]))]
    for t in range(T - 2, -1, -1):
        path.insert(0, int(backpointer[path[0], t]))
    return path, np.max(delta[:, T - 1])


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
