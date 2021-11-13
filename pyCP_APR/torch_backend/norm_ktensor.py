#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of norm utility with Numpy backend from the MATLAB Tensor Toolbox [1].
References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import torch as tr
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

    weightsMatrix = tr.ones([M.Rank, M.Rank]).to(M.device) * M.Weights
    coefMatrix = weightsMatrix * weightsMatrix.T

    for d in range(M.Dimensions):
        tmp = tr.matmul(M.Factors[str(d)].T, M.Factors[str(d)])
        coefMatrix = tr.mul(coefMatrix, tmp)

    nrm = sqrt(tr.abs(tr.sum(coefMatrix[:])))

    return nrm