#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Takes the Khatrirao product of KRUSKAL tensor M.

References
========================================
[1] Mrdmnd. (n.d.). mrdmnd/scikit-tensor. GitHub. https://github.com/mrdmnd/scikit-tensor/blob/master/src/tensor_tools.py.
"""

import numpy as np

def khatrirao(M_, dims, reverse=True):
    """
    KHATRIRAO Khatri-Rao product of matrices.
    **Citation:**
    Mrdmnd. (n.d.). mrdmnd/scikit-tensor. GitHub. https://github.com/mrdmnd/scikit-tensor/blob/master/src/tensor_tools.py.

    Parameters
    ----------
    M : object
    KRUSKAL tensor class. ktensor.K_TENSOR.
    dims : list
        which modes to multiply.
    reverse : bool, optional
        When true, product is in reverse order. The default is True.

    Raises
    ------
    ValueError
        Invalid tensors.

    Returns
    -------
    P : array
        Khatri-Rao product of matrices.

    """


    matrices = list()
    for d in dims:
        matrices.append(M_.Factors[d])

    matorder = range(len(matrices)) if not reverse else list(reversed(range(len(matrices))))
    N = matrices[0].shape[1]

    M = 1
    for i in matorder:
        if matrices[i].ndim != 2:
            raise ValueError("Each argument must be a matrix.")
        if N != (matrices[i].shape)[1]:
            raise ValueError("All matrices must have the same number of columns.")
        M *= (matrices[i].shape)[0]
        
    P = np.zeros((M, N))

    for n in range(N):
        ab = matrices[matorder[0]][:, n]

        for i in matorder[1:]:
            ab = np.outer(matrices[i][:, n], ab[:])
        P[:, n] = ab.flatten()

    return P
