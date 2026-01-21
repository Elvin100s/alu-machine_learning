#!/usr/bin/env python3
"""Module for numpy matrix multiplication"""
import numpy as np


def np_matmul(mat1, mat2):
    """Perform matrix multiplication on numpy arrays
    
    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray
        
    Returns:
        A new numpy.ndarray that is the matrix product
    """
    return np.matmul(mat1, mat2)
