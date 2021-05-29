#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor.py contains the TENSOR class for tensor X object representation.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
import copy
import numpy as np


class TENSOR():

    def __init__(self, Tensor):
        """
        Initilize the tensor X class.\n
        Creates the object representation of X.

        Parameters
        ----------
        Tensor : array
            Dense Numpy tensor.

        Returns
        -------
        None.

        """

        self.Size = list(Tensor.shape)
        self.Dimensions = Tensor.ndim
        self.Tensor = Tensor
        self.Type = 'tensor'

    def permute(self, order):
        """
        This function permutes the dimensions of X.

        Parameters
        ----------
        order : array
            Vector order.

        """
        self.Tensor = np.transpose(self.Tensor, order)

    def ttv(self, vecs):
        """
        Tensor times vector for KRUSKAL tensor M.

        Parameters
        ----------
        vecs : array
            coluumn vector.

        Returns
        -------
        c : array
             product of KRUSKAL tensor X with a (column) vector vecs.

        """

        dims = np.arange(self.Dimensions)
        vidx = np.arange(self.Dimensions)

        remdims = np.setdiff1d(dims, vidx)

        if self.Dimensions > 0:
            self.permute([x for x in range(0, self.Dimensions)])

        c = copy.deepcopy(self.Tensor)

        n = self.Dimensions
        for ii in [x for x in reversed(range(0, self.Dimensions))]:
            x = int(np.prod(self.Size[0:n - 1]))
            y = self.Size[n - 1]

            c = np.reshape(c, [x, y])
            c = np.dot(c, vecs[str(ii)])
            n -= 1

        return c
