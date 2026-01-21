#!/usr/bin/env python3
"""Module for matrix multiplication"""


def mat_mul(mat1, mat2):
    """Perform matrix multiplication
    
    Args:
        mat1: First 2D matrix
        mat2: Second 2D matrix
        
    Returns:
        A new matrix that is the product, or None if incompatible
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
             for j in range(len(mat2[0]))]
            for i in range(len(mat1))]
