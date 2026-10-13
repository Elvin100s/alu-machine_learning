#!/usr/bin/env python3
"""
Defines function that calculates the maximization step in the EM algorithm
for a Gaussian Mixture Model
"""


import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        g [numpy.ndarray of shape (k, n)]:
            containing the posterior probabilities for each data point
                in the cluster

    should only use one loop

    returns:
        pi, m, S:
            pi [numpy.ndarray of shape (k,)]:
                containing the updated priors for each cluster
            m [numpy.ndarray of shape (k, d)]:
                containing the updated centroid means for each cluster
            S [numpy.ndarray of shape (k, d, d)]:
                containing the updated covariance matrices for each cluster
        or None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    n, d = X.shape
    if type(g) is not np.ndarray or len(g.shape) != 2 or g.shape[1] != n:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None
    k = g.shape[0]
    n_k = np.sum(g, axis=1)
    pi = n_k / n
    m = np.matmul(g, X) / n_k[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / n_k[i]
    return pi, m, S
