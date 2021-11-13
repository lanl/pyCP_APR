#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of khatrirao utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

import numpy as np
from . ttv_sptensor import ttv

def khatrirao(X, M, n):
    """
    Takes the Khatrirao product of sparse tensor X and KRUSKAL tesor M
    
    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    X : sptensor tensor
        Sparse tensor X.
    n : int
        Mode to skip
    
    Returns
    -------
    K : array
        KRUSKAL tensor where signs on latent factor columns have been flipped.

    """
    N = X.Dimensions
    
    if n == 0:
        R = M.Factors["1"].shape[1]
    else:
        R = M.Factors["0"].shape[1]
    
    V = np.zeros((X.Size[n], R))
    
    for r in range(R):
        Z = {}
        for d in range(N):
            Z[str(d)] = []
        
        target_dimensions = list(np.arange(0, N))
        target_dimensions.pop(target_dimensions.index(n))
        for i in target_dimensions:
            Z[str(i)] = M.Factors[str(i)][:,r]
            
        ttv_res = ttv(X, Z, -n)
        V[:,r] = ttv_res.data
        
    return V