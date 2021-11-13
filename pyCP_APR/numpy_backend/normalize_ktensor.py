#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of normalize utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import numpy as np

def normalize(M, normtype=1, N=-1, mode=-1):
    """
    This function normalizes the columns of the factor matrices.

    Parameters
    ----------
    M : object
        KRUSKAL tensor M class. ktensor.K_TENSOR.
    normtype : int, optional
        Determines the type of normalization. The default is 1.
    N : int, optional
        Factor matrix number. The default is -1.
    mode : int, optional
        Dimension number. The default is -1.

    Returns
    -------
    M : object
        Normalized KRUSKAL tensor M class. ktensor.K_TENSOR.

    """
    # If the target dimension is given
    if mode != -1:
        for r in range(M.Rank):

            tmp = np.linalg.norm(M.Factors[str(mode)][:, r], normtype)

            if tmp > 0:
                M.Factors[str(mode)][:, r] /= tmp

            M.Weights[r] *= tmp

        return M

    # Normalize each of the component and weights
    for r in range(M.Rank):
        for d in range(M.Dimensions):

            tmp = np.linalg.norm(M.Factors[str(d)][:, r], normtype)

            if tmp > 0:
                M.Factors[str(d)][:, r] /= tmp

            M.Weights[r] *= tmp

    negative_components = np.where(M.Weights < 0)
    M.Factors[str(0)][:, negative_components] *= -1
    M.Weights[negative_components] *= -1

    # Absorb the weight into one factor
    if N == 0:
        D = np.diag(np.power(M.Weights, 1 / M.Dimensions))
        for dim in range(M.Dimensions):
            M.Factors[str(dim)] = np.dot(M.Factors[str(dim)], D)
        M.Weights = np.ones(M.Rank)

    elif N > 0:
        M.Factors[str(N - 1)] = np.dot(M.Factors[str(N - 1)], np.diag(M.Weights))
        M.Weights = np.ones(M.Rank)

    elif N == -2:
        
        if M.Rank > 1:
            p = np.argsort(-M.Weights)
            M = arrange(M, p)

    return M


def arrange(M, p=[]):
    """
    This function arranges the components of KRUSKAL tensor M.

    Parameters
    ----------
    p : list, optional
        permutation. The default is [].

    """
    # Just rearrange and return if second argument is a permutation
    if len(p) > 0:
        
        M.Weights = M.Weights[p]
        for d in range(M.Dimensions):
            M.Factors[str(d)] = M.Factors[str(d)][:, p]
        
        return M
    
    # Ensure that matrices are normalized
    M = normalize(M)
    
    # sort
    idx = np.argsort(-M.Weights)
    M.Weights = M.Weights[idx]
    
    for d in range(M.Dimensions):
        M.Factors[str(d)] = M.Factors[str(d)][:,idx]
        
    return M