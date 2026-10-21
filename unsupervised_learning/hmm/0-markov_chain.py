#!/usr/bin/env python3
"""Computes the state probabilities of a Markov chain after t iterations."""
import numpy as np


def markov_chain(P, s, t=1):
    """Determine the probability of a Markov chain being in a state after t.

    Args:
        P (numpy.ndarray): square transition matrix of shape (n, n).
        s (numpy.ndarray): starting probabilities of shape (1, n).
        t (int): number of iterations the chain has been through.

    Returns:
        numpy.ndarray: probabilities of shape (1, n) after t iterations,
        or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if n == 0 or P.shape[1] != n:
        return None
    if not _is_numeric(P) or (P < 0).any() or (P > 1).any():
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if not _is_numeric(s) or (s < 0).any() or (s > 1).any():
        return None
    if not np.isclose(np.sum(s), 1):
        return None
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0:
        return None

    state = s.copy()
    for _ in range(t):
        state = np.matmul(state, P)
    return state


def _is_numeric(matrix):
    """Check that an array holds real numbers.

    Args:
        matrix (numpy.ndarray): array to check.

    Returns:
        bool: True if the array holds finite real numbers, False otherwise.
    """
    return bool(np.issubdtype(matrix.dtype, np.number) and
                not np.iscomplexobj(matrix) and np.all(np.isfinite(matrix)))
