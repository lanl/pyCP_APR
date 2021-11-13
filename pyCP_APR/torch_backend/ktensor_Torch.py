#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ktensor_Torch.py contains the K_TENSOR class for KRUSKAL tensor M object representation.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
from math import sqrt
import sys
import torch as tr


class K_TENSOR():

    def __init__(self, Rank, Size, Minit='random', random_state=42, device='cpu', dtype='torch.DoubleTensor'):
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
        device : string, optional
            Torch device to be used.
            'cpu' to use PyTorch with CPU.
            'gpu' to use cuda:0
            The default is cpu.
        dtype : string, optional
            Type to be used in torch tensors.
            Default is torch.cuda.DoubleTensor.

        """

        self.Factors = dict()
        self.device = device
        self.dtype = dtype
        self.Weights = tr.ones(Rank).to(self.device)
        self.Rank = Rank
        self.Dimensions = len(Size)
        self.Type = 'ktensor'

        if Minit == 'random':
            tr.random.manual_seed(random_state)
            for d in range(self.Dimensions):
                if self.dtype == 'torch.FloatTensor':
                    self.Factors[str(d)] = tr.FloatTensor(Size[d], Rank).uniform_(0, 1).to(self.device)
                else:
                    self.Factors[str(d)] = tr.DoubleTensor(Size[d], Rank).uniform_(0, 1).to(self.device)
        # if initial Factors are passed
        else:
            for d in range(self.Dimensions):
                if "Factors" in Minit:
                    self.Factors[str(d)] = Minit["Factors"][str(d)].to(self.device)
                else:
                    self.Factors[str(d)] = Minit[str(d)].to(self.device)
                    
            # initial weights are passed
            if "Weights" in Minit:
                if len(Minit["Weights"]) > self.Rank:
                    raise Exception("Number of weights must be same as the tensor rank!")
                
                if isinstance(Minit["Weights"], (list, np.ndarray)):
                    self.Weights = tr.from_numpy(Minit["Weights"]).type(self.dtype).to(self.device)
                else:
                    self.Weights = Minit["Weights"].to(self.device)

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
            Factors_[str(d)] = tr.copy(self.Factors[str(d)])

        return Factors_

