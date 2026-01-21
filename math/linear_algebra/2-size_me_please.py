#!/usr/bin/env python3
"""Module for calculating matrix shape"""


def matrix_shape(matrix):
    """Calculate the shape of a matrix
    
    Args:
        matrix: A nested list representing a matrix
        
    Returns:
        A list of integers representing the shape
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
