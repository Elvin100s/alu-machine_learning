#!/usr/bin/env python3
"""
Defines function that calculates the probability density function of
a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset whose PDF should be calculated
            n: the number of data points
            d: the number of dimensions for each data point
        m [numpy.ndarray of shape (d,)]:
            contains the mean of the distribution
        S [numpy.ndarray of shape (d, d)]:
            contains the covariance of the distribution

    not allowed to use any loops
    not allowed to use the function numpy.diag or method numpy.ndarray.diagonal

    returns:
        P [numpy.ndarray of shape (n,)]:
            containing the PDF values for each data point
            all values in P should have a minimum value of 1e-300
        or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    d = X.shape[1]
    if type(m) is not np.ndarray or m.shape != (d,):
        return None
    if type(S) is not np.ndarray or S.shape != (d, d):
        return None
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    diff = X - m
    # Mahalanobis term for each data point
    maha = np.sum(np.matmul(diff, inv) * diff, axis=1)
    coef = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    P = coef * np.exp(-maha / 2)
    return np.maximum(P, 1e-300)
