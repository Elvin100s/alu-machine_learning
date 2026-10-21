#!/usr/bin/env python3
"""Computes the steady state probabilities of a regular Markov chain."""
import numpy as np


def is_regular(P):
    """Determine whether a transition matrix is regular.

    A Markov chain is regular when some power of its transition matrix
    contains only positive entries.

    Args:
        P (numpy.ndarray): square transition matrix of shape (n, n).

    Returns:
        bool: True if the chain is regular, False otherwise.
    """
    n = P.shape[0]
    power = P.copy()
    for _ in range((n - 1) ** 2 + 1):
        if np.all(power > 0):
            return True
        power = np.matmul(power, P)
    return False


def regular(P):
    """Determine the steady state probabilities of a regular Markov chain.

    Args:
        P (numpy.ndarray): square transition matrix of shape (n, n).

    Returns:
        numpy.ndarray: steady state probabilities of shape (1, n),
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
    if not is_regular(P):
        return None

    # left eigenvector of eigenvalue 1, normalized so that it sums to 1
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    index = int(np.argmin(np.abs(eigenvalues - 1)))
    if not np.isclose(eigenvalues[index], 1):
        return None
    steady = np.real(eigenvectors[:, index])
    if np.sum(steady) < 0:
        steady = -steady
    return (steady / np.sum(steady)).reshape(1, n)


def _is_numeric(matrix):
    """Check that an array holds real numbers.

    Args:
        matrix (numpy.ndarray): array to check.

    Returns:
        bool: True if the array holds finite real numbers, False otherwise.
    """
    return bool(np.issubdtype(matrix.dtype, np.number) and
                not np.iscomplexobj(matrix) and np.all(np.isfinite(matrix)))
