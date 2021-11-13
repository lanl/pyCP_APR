#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of tt_dimscheck utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import numpy as np

def tt_dimscheck(dims, N, M):
    """
    Processes tensor dimensions.

    Parameters
    ----------
    dims : list or int
        Dimension indices.
    N : int
        tensor order.
    M : int
        Multiplicants
        
    Returns
    -------
    sdims : list
        index for M muliplicands
    vidx : list
        index for M muliplicands
    """
    
    if isinstance(dims, list) or isinstance(dims, np.ndarray):
        if len(dims) > 0:
            if (max(dims) < 0):
                dims = np.setdiff1d(np.arange(0, N), -dims)
        else:
            dims = np.arange(0, N)
    else:
        if dims < 0 or dims == 0:
            dims = np.setdiff1d(np.arange(0, N), -dims)
    
    # Save the number of dimensions in dims
    P = len(dims)
    
    # Reorder dims from smallest to largest (this matters in particular
    # for the vector multiplicand case, where the order affects the result)
    sidx = np.argsort(dims)
    sdims = np.array(dims)[sidx]
    
    # Check sizes to determine how to index multiplicands
    if P == M:
        # Case 1: Number of items in dims and number of multiplicands
        # are equal; therefore, index in order of how sdims was sorted.
        vidx = sidx
    else:
        # Case 2: Number of multiplicands is equal to the number of
        # dimensions in the tensor; therefore, index multiplicands by
        # dimensions specified in dims argument.
        vidx = sdims
        
    return sdims, vidx