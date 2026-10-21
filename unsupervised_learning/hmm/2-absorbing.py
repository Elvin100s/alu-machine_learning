#!/usr/bin/env python3
"""Determines whether a Markov chain is absorbing."""
import numpy as np


def absorbing(P):
    """Determine whether a Markov chain is absorbing.

    A chain is absorbing when every state can reach at least one absorbing
    state, an absorbing state being a state that only transitions to itself.

    Args:
        P (numpy.ndarray): standard transition matrix of shape (n, n).

    Returns:
        bool: True if the chain is absorbing, False otherwise.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if n == 0 or P.shape[1] != n:
        return False
    if not _is_numeric(P) or (P < 0).any() or (P > 1).any():
        return False

    absorbing_states = [i for i in range(n) if P[i, i] == 1]
    if not absorbing_states:
        return False

    # every state must reach an absorbing state through positive transitions
    reached = np.zeros(n, dtype=bool)
    reached[absorbing_states] = True
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if reached[i]:
                continue
            if np.any(P[i][reached]):
                reached[i] = True
                changed = True
    return bool(np.all(reached))


def _is_numeric(matrix):
    """Check that an array holds real numbers.

    Args:
        matrix (numpy.ndarray): array to check.

    Returns:
        bool: True if the array holds finite real numbers, False otherwise.
    """
    return bool(np.issubdtype(matrix.dtype, np.number) and
                not np.iscomplexobj(matrix) and np.all(np.isfinite(matrix)))
