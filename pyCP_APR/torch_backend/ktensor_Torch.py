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
                self.Factors[str(d)] = Minit[str(d)].to(self.device)

    def innerprod(self, X):
        """
        This function takes the inner product of tensor X and KRUSKAL tensor M.

        Parameters
        ----------
        X : class
            Original tensor. sptensor_Torch.SP_TENSOR.

        Returns
        -------
        res : array
            inner product of tensor X and KRUSKAL tensor M.

        """

        # if there are no nonzero terms in X.
        if len(X.Values) == 0:
            res = 0

        vecs = dict()
        res = 0

        for r in range(self.Rank):
            for d in range(self.Dimensions):
                vecs[str(d)] = self.Factors[str(d)][:, r]
            res = res + self.Weights[r] * X.ttv(vecs)

        return res

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

    def norm(self):
        """
        This function takes the Frobenius norm of a KRUSKAL tensor M.

        Returns
        -------
        nrm : float
            Frobenius norm of M.

        """

        weightsMatrix = tr.ones([self.Rank, self.Rank]).to(self.device) * self.Weights
        coefMatrix = weightsMatrix * weightsMatrix.T

        for d in range(self.Dimensions):
            tmp = tr.matmul(self.Factors[str(d)].T, self.Factors[str(d)])
            coefMatrix = tr.mul(coefMatrix, tmp)

        nrm = sqrt(tr.abs(tr.sum(coefMatrix[:])))

        return nrm

    def redistribute(self, mode):
        """
        This function distributes the weights to a specified dimension or mode.\n

        Parameters
        ----------
        mode : int
            Dimension number.

        """

        for r in range(self.Rank):
            self.Factors[str(mode)][:, r] *= self.Weights[r]
            self.Weights[r] = 1

    def arrange(self, p=[]):
        """
        This function arranges the components of KRUSKAL tensor M.

        Parameters
        ----------
        p : list, optional
            permutation. The default is [].

        """

        # Just rearrange and return if second argument is a permutation
        if len(p) > 0:

            self.Weights = self.Weights[p]
            for d in range(self.Dimensions):
                self.Factors[str(d)] = self.Factors[str(d)][:, p]

    def normalize(self, M, normtype=1, N=-1, mode=-1):
        """
        This function normalizes the columns of the factor matrices.

        Parameters
        ----------
        M : object
            KRUSKAL tensor M class. ktensor_Torch.K_TENSOR.
        normtype : int, optional
            Determines the type of normalization. The default is 1.
        N : int, optional
            Factor matrix number. The default is -1.
        mode : int, optional
            Dimension number. The default is -1.

        Returns
        -------
        M : object
            Normalized KRUSKAL tensor M class. ktensor_Torch.K_TENSOR.

        """

        # If the target dimension is given
        if mode != -1:
            for r in range(M.Rank):

                tmp = tr.norm(M.Factors[str(mode)][:, r], normtype)

                if tmp > 0:
                    M.Factors[str(mode)][:, r] /= tmp

                M.Weights[r] *= tmp

            return M

        # Normalize each of the component and weights
        for r in range(M.Rank):
            for d in range(M.Dimensions):

                tmp = tr.norm(M.Factors[str(d)][:, r], normtype)

                if tmp > 0:
                    M.Factors[str(d)][:, r] /= tmp

                M.Weights[r] *= tmp

        negative_components = tr.where(M.Weights < 0)

        M.Factors[str(0)][:, [t.to('cpu').numpy() for t in negative_components]] *= -1
        M.Weights[negative_components] *= -1

        # Absorb the weight into one factor
        if N == 0:
            sys.exit("Reached to a location that has not been imlemented yet.")

        elif N > 0:
            M.Factors[str(N - 1)] = tr.matmul(M.Factors[str(N - 1)], tr.diag(M.Weights).to(self.device))
            Lambdas = tr.ones(M.Rank)

        elif N == -2:
            if M.Rank > 1:
                p = tr.argsort(-M.Weights)
                M.arrange(p)

        return M
