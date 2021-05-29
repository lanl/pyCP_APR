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

    def __init__(self, Rank, Size, Minit='random', random_state=42, order=-1, weights=-1):
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
        weights : array, optional
            Initial weights of the components.\n
            If not passed, initial weights are 1.\n
            The default is -1.

        """

        self.Factors = dict()
        self.Rank = Rank
        self.Dimensions = len(Size)
        self.Size = Size
        self.Type = 'ktensor'

        # If the initial weights are passed
        if isinstance(weights, (np.ndarray)):
            self.Weights = weights

        # Assign the component weights
        else:
            self.Weights = np.ones(Rank)

        if Minit == 'random':

            np.random.seed(random_state)
            for d in range(self.Dimensions):
                self.Factors[str(d)] = np.random.uniform(low=0, high=1, \
                                                         size=(Size[d], Rank))
        # if initial Factors are passed
        else:
            for d in range(self.Dimensions):
                self.Factors[str(d)] = Minit[str(d)]

    def double(self):
        """
        This function converts the KTENSOR M to a double array.

        Returns
        -------
        A : array
            Double array of M.

        """
        sz = self.Size
        # A = X.lambda' * khatrirao(X.u,'r')'
        nn = [str(x) for x in reversed(range(0, self.Dimensions))]

        A = np.dot(self.Weights.T, self.khatrirao(nn).T)
        A = np.reshape(A, sz)

        return A

    def permute(self, order):
        """
        This function permutes the dimensions of the KRUSKAL tensor M.

        Parameters
        ----------
        order : array
            Vector order.

        """

        for ii, dim in enumerate(order):
            temp = self.Factors[str(ii)]
            self.Factors[str(ii)] = self.Factors[str(dim)]
            self.Factors[str(dim)] = temp

    def khatrirao(self, dims, reverse=True):
        """
        KHATRIRAO Khatri-Rao product of matrices.
        **Citation:**
        Mrdmnd. (n.d.). mrdmnd/scikit-tensor. GitHub. https://github.com/mrdmnd/scikit-tensor/blob/master/src/tensor_tools.py.

        Parameters
        ----------
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
            matrices.append(self.Factors[d])

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

    def innerprod(self, X):
        """
        This function takes the inner product of tensor X and KRUSKAL tensor M.

        Parameters
        ----------
        X : class
            Original tensor. sptensor.SP_TENSOR.

        Returns
        -------
        res : array
            inner product of tensor X and KRUSKAL tensor M.

        """
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
            Factors_[str(d)] = np.copy(self.Factors[str(d)])

        return Factors_

    def norm(self):
        """
        This function takes the Frobenius norm of a KRUSKAL tensor M.

        Returns
        -------
        nrm : float
            Frobenius norm of M.

        """
        weightsMatrix = np.ones([self.Rank, self.Rank]) * self.Weights
        coefMatrix = weightsMatrix * weightsMatrix.T

        for d in range(self.Dimensions):
            tmp = np.dot(self.Factors[str(d)].T, self.Factors[str(d)])
            coefMatrix = np.multiply(coefMatrix, tmp)

        nrm = sqrt(np.abs(np.sum(coefMatrix[:])))

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
            KRUSKAL tensor M class. ktensor.K_TENSOR.
        normtype : int, optional
            Determines the type of normalization. The default is 1.
        N : int, optional
            Factor matrix number. The default is -1.
        mode : int, optional
            Dimension number. The default is -1.

        Returns
        -------
        M : object
            Normalized KRUSKAL tensor M class. ktensor.K_TENSOR.

        """
        # If the target dimension is given
        if mode != -1:
            for r in range(M.Rank):

                tmp = np.linalg.norm(M.Factors[str(mode)][:, r], normtype)

                if tmp > 0:
                    M.Factors[str(mode)][:, r] /= tmp

                M.Weights[r] *= tmp

            return M

        # Normalize each of the component and weights
        for r in range(M.Rank):
            for d in range(M.Dimensions):

                tmp = np.linalg.norm(M.Factors[str(d)][:, r], normtype)

                if tmp > 0:
                    M.Factors[str(d)][:, r] /= tmp

                M.Weights[r] *= tmp

        negative_components = np.where(M.Weights < 0)
        M.Factors[str(0)][:, negative_components] *= -1
        M.Weights[negative_components] *= -1

        # Absorb the weight into one factor
        if N == 0:
            D = np.diag(np.power(M.Weights, 1 / M.Dimensions))
            for dim in range(M.Dimensions):
                M.Factors[str(dim)] = np.dot(M.Factors[str(dim)], D)
            M.Weights = np.ones(M.Rank)

        elif N > 0:
            M.Factors[str(N - 1)] = np.dot(M.Factors[str(N - 1)], np.diag(M.Weights))
            Lambdas = np.ones(M.Rank)

        elif N == -2:
            if M.Rank > 1:
                p = np.argsort(-M.Weights)
                M.arrange(p)

        return M
