#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of redistribute utility with Numpy backend from the MATLAB Tensor Toolbox [1].
References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

def redistribute(M, mode):
    """
    This function distributes the weights to a specified dimension or mode.\n
    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    mode : int
        Dimension number.
        
    Returns
    -------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    """
    for r in range(M.Rank):
        M.Factors[str(mode)][:, r] *= M.Weights[r]
        M.Weights[r] = 1

    return M