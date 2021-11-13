#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of arrange utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

import numpy as np
from . normalize_ktensor import normalize

def arrange(M, p=[]):
    """
    This function arranges the components of KRUSKAL tensor M.

    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    p : list, optional
        permutation. The default is [].
        
    Returns
    -------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.

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