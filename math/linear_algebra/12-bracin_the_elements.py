#!/usr/bin/env python3
"""Module for numpy element-wise operations"""


def np_elementwise(mat1, mat2):
    """Perform element-wise operations on numpy arrays
    
    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray or scalar
        
    Returns:
        A tuple of (sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
