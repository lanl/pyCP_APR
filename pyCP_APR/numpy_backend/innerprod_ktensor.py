#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of innerprod utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

from . ttv_tensor import ttv as ttvt
from . ttv_sptensor import ttv as ttvsp

def innerprod(M, X):
    """
    This function takes the inner product of tensor X and KRUSKAL tensor M.

    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    X : class
        Original tensor. sptensor.SP_TENSOR.

    Returns
    -------
    res : array
        inner product of tensor X and KRUSKAL tensor M.

    """
    if X.Type == "sptensor":
        ttv = ttvsp
    elif X.Type == "tensor":
        ttv = ttvt
    else:
        raise Exception("Unkown tensor type!")

    vecs = dict()
    res = 0

    for r in range(M.Rank):
        for d in range(M.Dimensions):
            vecs[str(d)] = M.Factors[str(d)][:, r]
        res = res + M.Weights[r] * ttv(X, vecs).data

    return res