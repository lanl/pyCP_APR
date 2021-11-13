#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of norm utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import numpy as np
from math import sqrt

def norm(M):
    """
    This function takes the Frobenius norm of a KRUSKAL tensor M.
    
    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.

    Returns
    -------
    nrm : float
        Frobenius norm of M.

    """
    weightsMatrix = np.ones([M.Rank, M.Rank]) * M.Weights
    coefMatrix = weightsMatrix * weightsMatrix.T

    for d in range(M.Dimensions):
        tmp = np.dot(M.Factors[str(d)].T, M.Factors[str(d)])
        coefMatrix = np.multiply(coefMatrix, tmp)

    nrm = sqrt(np.abs(np.sum(coefMatrix[:])))

    return nrm