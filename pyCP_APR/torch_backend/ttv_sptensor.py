#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of normalize utility with Numpy backend from the MATLAB Tensor Toolbox [1].
References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import torch as tr
from . tt_dimscheck import tt_dimscheck

def ttv(M, vecs, dims=[]):
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

    dims = tr.arange(M.Dimensions)
    vidx = tr.arange(M.Dimensions)

    combined = tr.cat((dims, vidx))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]

    remdims = difference

    for d in range(M.Dimensions):
        if len(vecs[str(vidx.data.tolist()[d])]) != M.Size[d]:
            sys.exit('Multiplicand is wrong size')

    newvals = M.data
    subs = M.Coords

    if len(subs) == 0:
        newsubs = []

    for d in range(M.Dimensions):
        idx = M.Coords[:, d]
        w = vecs[str(vidx.data.tolist()[d])]
        bigw = w[idx]
        newvals = tr.mul(newvals, bigw)

    newsubs = subs[:, remdims]

    if len(remdims) == 0:
        c = tr.sum(newvals)
        return c

    raise Exception("Reached to a location that has not been imlemented yet.")
    return -1