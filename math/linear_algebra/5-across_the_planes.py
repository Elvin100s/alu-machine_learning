#!/usr/bin/env python3
"""Module for 2D matrix addition"""


def add_matrices2D(mat1, mat2):
    """Add two 2D matrices element-wise
    
    Args:
        mat1: First 2D matrix
        mat2: Second 2D matrix
        
    Returns:
        A new matrix with element-wise sum, or None if shapes don't match
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
