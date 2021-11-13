#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of fixsigns_oneargin utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import numpy as np

def fixsigns_oneargin(K):
    """
    Fix sign ambiguity of a ktensor.
    
    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    
    Returns
    -------
    K : array
        KRUSKAL tensor where signs on latent factor columns have been flipped.

    """
    R = K.Rank
    
    val = [0]*K.Dimensions
    idx = [0]*K.Dimensions
    sgn = [0]*K.Dimensions
    for r in range(R):
        for n in range(K.Dimensions):
            idx[n] = np.argmax(np.abs(K.Factors[str(n)][:,r]))
            val[n] = np.max(np.abs(K.Factors[str(n)][:,r]))
            sgn[n] = np.sign(K.Factors[str(n)][idx[n],r])
            
        negidx = np.argwhere(sgn == -1)
        nflip = int(2 * np.floor(len(negidx)/2))
        
        for i in range(nflip):
            n = negidx[i]
            K.Factors[str(n)][:,r] = -K.Factors[str(n)][:,r]
        
    return K