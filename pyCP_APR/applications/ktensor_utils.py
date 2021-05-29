#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ktensor_utils.py contains the utility functions for KRUSKAL tensor M.

@author: Maksim Ekin Eren
"""
import numpy as np

def get_X_hat(components, indices):
    """
    Calculate X hat from KRUSKAL tensor M, given the non-zero indicies.\n

    components: KRUSKAL tensor components\n
    indices: non-zero coordinates

    Parameters
    ----------
    components : dict
        KRUSKAL Tensor M in dict format.
    indices : array
        Array of indices in X hat.

    Returns
    -------
    lambdas : array
        Array of lambdas in X calculated from M using the indices.

    """
    factors = components['Factors']
    gammas = components['Weights']

    if len(factors['0'].shape) == 1:

        return gammas * np.prod([factors[f'{ii}'][indices[:, ii]] for ii in range(indices.shape[1])], axis=0)
    else:
        return (gammas * np.prod([factors[f'{ii}'][indices[:, ii]] for ii in range(indices.shape[1])], axis=0)).sum(axis=1)

def get_X_size(components):
    """
    Get tensor shape from the components.

    Parameters
    ----------
    components : dict
        KRUSKAL Tensor M in dict format.

    Returns
    -------
    shape : list
        Tensor X shape.

    """
    if isinstance(components, (list, np.ndarray)):
        factors = components[0]['Factors']
    else:
        factors = components['Factors']
    return [factors[f'{i}'].shape[0] for i in range(len(factors))]
