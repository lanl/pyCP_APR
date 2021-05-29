#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tenmat.py creates a matricized tensor.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
import copy

import numpy as np

from . ktensor import K_TENSOR
from . tensor import TENSOR


class Tenmat():

    def __init__(self, X, mode):
        """
        Create a matricized tensor.

        Parameters
        ----------
        X : class
            Original tensor. sptensor.SP_TENSOR or tensor.TENSOR
        mode : int
            Dimension number.

        """

        if X.Type == 'tensor':
            T = TENSOR(copy.deepcopy(X.Tensor))


        elif X.Type == 'ktensor':
            T = K_TENSOR(X.Rank, X.Size, Minit=X.Factors)

        rdims = [mode]
        tmp = [True] * T.Dimensions
        tmp[rdims[0]] = False
        cdims = np.where(tmp)[0]
        order = rdims + list(cdims)

        T.permute(order)

        x = np.prod([T.Size[i] for i in rdims])
        y = np.prod([T.Size[i] for i in cdims])

        if X.Type == 'tensor':
            self.Tensor = np.reshape(T.Tensor, [x, y])

        elif X.Type == 'ktensor':
            A = T.double()
            self.Tensor = np.reshape(A, [x, y])
