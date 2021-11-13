#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ktensor.py contains the K_TENSOR class for KRUSKAL tensor M object representation.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
from math import sqrt
import numpy as np


class K_TENSOR():

    def __init__(self, Rank, Size, Minit='random', random_state=42, order=-1):
        """
        Initilize the K_TENSOR class.\n
        Creates the object representation of M.\n
        If initial M is not passed, by default, creates M from uniform distribution.

        Parameters
        ----------
        Rank : int
            Tensor rank, i.e. number of components in M.
        Size : list
            Shape of the tensor.
        Minit : string or dictionary of latent factors
            Initial value of latent factors.\n
            If Minit = 'random', initial factors are chosen randomly from uniform distribution between 0 and 1.\n
            Else, pass dictionary where the key is the mode number and value is array size d x r
            where d is the number of elements on the dimension and r is the rank.\n
            The default is "random".
        random_state : int, optional
            Random seed for initial M.
            The default is 42.
        order : int, optional
            Currently not used. The default is -1.

        """

        self.Factors = dict()
        self.Rank = Rank
        self.Dimensions = len(Size)
        self.Size = Size
        self.Type = 'ktensor'
        self.Weights = np.ones(Rank)

        if Minit == 'random':

            np.random.seed(random_state)
            for d in range(self.Dimensions):
                self.Factors[str(d)] = np.random.uniform(low=0, high=1, \
                                                         size=(Size[d], Rank))
        # if initial Factors are passed
        else:
            for d in range(self.Dimensions):
                if "Factors" in Minit:
                    self.Factors[str(d)] = Minit["Factors"][str(d)]
                else:
                    self.Factors[str(d)] = Minit[str(d)]
            
            # initial weights are passed
            if "Weights" in Minit:
                if len(Minit["Weights"]) > self.Rank:
                    raise Exception("Number of weights must be same as the tensor rank!")
                self.Weights = np.array(Minit["Weights"], dtype='f')

    def deep_copy_factors(self):
        """
        Creates a deep copy of the latent factors in M.

        Returns
        -------
        factors : dict
            Copy of the latent factors of M.

        """
        # create a copy of the current factors
        Factors_ = dict()
        for d in range(self.Dimensions):
            Factors_[str(d)] = np.copy(self.Factors[str(d)])

        return Factors_
