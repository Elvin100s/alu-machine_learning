#!/usr/bin/env python3
"""Module for matrix transpose"""


def matrix_transpose(matrix):
    """Return the transpose of a 2D matrix
    
    Args:
        matrix: A 2D list representing a matrix
        
    Returns:
        A new matrix that is the transpose
    """
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
