#!/usr/bin/env python3
"""
Defines function that calculates the expectation step in the EM algorithm
for a Gaussian Mixture Model
"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        pi [numpy.ndarray of shape (k,)]:
            contains the priors for each cluster
        m [numpy.ndarray of shape (k, d)]:
            contains the centroid means for each clustern
        S [numpy.ndarray of shape (k, d, d)]:
            contains the covariance matrices for each cluster

    should only use one loop

    returns:
        g, l:
            g [numpy.ndarray of shape (k, n)]:
                containing the posterior probabilities for each data point
                    in the cluster
            l [float]:
                total log likelihood
        or None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    k = pi.shape[0]
    if type(m) is not np.ndarray or m.shape != (k, d):
        return None, None
    if type(S) is not np.ndarray or S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None
    num = np.zeros((k, n))
    for i in range(k):
        num[i] = pi[i] * pdf(X, m[i], S[i])
    marginal = np.sum(num, axis=0)
    g = num / marginal
    log_l = np.sum(np.log(marginal))
    return g, log_l
