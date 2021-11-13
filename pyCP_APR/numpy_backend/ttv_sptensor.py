#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of ttv utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

import numpy as np
from . tt_dimscheck import tt_dimscheck
from . sptensor import SP_TENSOR
from . tensor import TENSOR
from . accum import accum as accumarray

def ttv(X, vecs, dims=[]):
    """
    Tensor times vector for KRUSKAL tensor M.

    Parameters
    ----------
    X : object
        Sparse tensor. sptensor.SP_TENSOR.
    vecs : array
        coluumn vector.
    dims : list
        list of dimension indices.

    Returns
    -------
    c : array
         product of KRUSKAL tensor X with a (column) vector vecs.

    """
    dims, vidx = tt_dimscheck(dims, X.Dimensions, len(vecs))
    remdims = np.setdiff1d(np.arange(X.Dimensions), vidx)

    for d in range(len(dims)):
        if vecs[str(vidx[d])].shape[0] != X.Size[dims[d]]:
            raise Exception('Multiplicand is wrong size')

    newvals = X.data
    subs = X.Coords

    if len(subs) == 0:
        newsubs = []

    for n in range(len(dims)):
        idx = X.Coords[:, dims[n]]
        w = vecs[str(vidx[n])]
        bigw = w[idx]
        newvals = np.multiply(newvals, bigw)

    newsubs = subs[:, remdims]
    
    # Case 0: If all dimensions were used, then just return the su
    if len(remdims) == 0:
        c = np.sum(newvals)
        return c
    
    # Otherwise, figure new subscripts and accumulate the results.
    newsiz = np.array(X.Size)[remdims]
    
    # Case I: Result is a vector
    if len(remdims) == 1:
        c = accumarray(accmap=newsubs, a=newvals, size=newsiz)
        
        if np.count_nonzero(c) <= .5 * newsiz[0]:
            c = SP_TENSOR(np.arange(0, newsiz).T, c, newsiz)
        else:
            c = TENSOR(c)
        
        return c
        
    # Case II: Result is a multiway array
    c = SP_TENSOR(newsubs, newvals, newsiz)
    
    return c
