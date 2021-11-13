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
        self.data = Tensor
        self.Type = 'tensor'
