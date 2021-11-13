#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of ttv utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

from . permute_tensor import permute
from . tt_dimscheck import tt_dimscheck
import numpy as np
import copy

def ttv(X, vecs):
    """
    Tensor times vector for KRUSKAL tensor M.

    Parameters
    ----------
    X : object
        Dense tensor class. tensor.TENSOR.
    vecs : array
        coluumn vector.

    Returns
    -------
    c : array
         product of KRUSKAL tensor X with a (column) vector vecs.

    """
    dims, vidx = tt_dimscheck(dims, X.Dimensions, len(vecs))
    remdims = np.setdiff1d(np.arange(X.Dimensions), vidx)

    if X.Dimensions > 0:
        X = permute(X, [x for x in range(0, X.Dimensions)])

    c = copy.deepcopy(X.data)

    n = X.Dimensions
    for ii in [x for x in reversed(range(0, X.Dimensions))]:
        x = int(np.prod(X.Size[0:n - 1]))
        y = X.Size[n - 1]

        c = np.reshape(c, [x, y])
        c = np.dot(c, vecs[str(ii)])
        n -= 1

    return c