#!/usr/bin/env python3
"""
Defines function that calculates total intra-cluster variance for a data set
"""


import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        C [numpy.ndarray of shape (k, d)]:
            contains the centroid means for each cluster
            k: the number of clusters
            d: the number of dimensions for each data point

    should not use any loops

    returns:
        var [float]: total variance
        or None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    # squared distance from every point to every centroid
    dist = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)
    # each point contributes the squared distance to its closest centroid
    var = np.sum(np.min(dist, axis=1))
    return var
