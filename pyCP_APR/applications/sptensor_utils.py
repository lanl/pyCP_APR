#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sptensor_utils.py contains the utility functions for tensor X.

@author: Maksim Ekin Eren
"""
import numpy as np


def get_X_dimensions(X):
    """
    Returns the number of dimensions that tensor X has.

    Parameters
    ----------
    X : array
        Tensor X in COO format. i.e. X is the coordinates of the non-zero values.

    Returns
    -------
    dimensions : int
        Number of dimensions that X has.

    """
    return X.shape[1]

def get_X_dim_size(X):
    """
    Returns the shape of X. i.e. size of each mode.

    Parameters
    ----------
    X : array
        Tensor X in COO format. i.e. X is the coordinates of the non-zero values.

    Returns
    -------
    size : int
        Tensor X shape.

    """
    size = list()
    dimensions = get_X_dimensions(X)

    for d in range(dimensions):
        size.append(np.amax(X[:, d]) + 1)

    return size

def get_X_size(X):
    """
    Calculates the total number of elements in X including non-zeros and zeros.

    Parameters
    ----------
    X : array
        Tensor X in COO format. i.e. X is the coordinates of the non-zero values.

    Returns
    -------
    size : int
        Number of elements in X.

    """
    X_dim_size = get_X_dim_size(X)
    return np.product(X_dim_size)

def get_X_num_non_zeros(X):
    """
    Calculates the number of non-zero elements in X.

    Parameters
    ----------
    X : array
        Tensor X in COO format. i.e. X is the coordinates of the non-zero values.

    Returns
    -------
    non-zeros : int
        Number of non-zeros in X.

    """
    return len(X)

def get_X_num_zeros(X):
    """
    Calculates the total number of zeros in X.

    Parameters
    ----------
    X : array
        Tensor X in COO format. i.e. X is the coordinates of the non-zero values.

    Returns
    -------
    zeros : int
        Number of zeros in X.

    """
    return get_X_size(X) - get_X_num_non_zeros(X)
