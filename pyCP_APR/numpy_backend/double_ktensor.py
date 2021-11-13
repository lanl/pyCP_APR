#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of double utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

import numpy as np
from . khatrirao_ktensor import khatrirao

def double(M):
    """
    This function converts the KTENSOR M to a double array.
    
    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    
    Returns
    -------
    A : array
        Double array of M.

    """
    sz = M.Size
    nn = [str(x) for x in reversed(range(0, M.Dimensions))]

    A = np.dot(M.Weights.T, khatrirao(M, nn).T)
    A = np.reshape(A, sz)

    return A