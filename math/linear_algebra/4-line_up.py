#!/usr/bin/env python3
"""Module for array addition"""


def add_arrays(arr1, arr2):
    """Add two arrays element-wise
    
    Args:
        arr1: First array
        arr2: Second array
        
    Returns:
        A new list with element-wise sum, or None if shapes don't match
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
