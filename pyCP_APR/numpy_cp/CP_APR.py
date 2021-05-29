#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of the CP-APR algorithm [1-4] with Numpy backend.\n
This backend can be used to factorize sparse tensorsin COO format and dense Numpy tensors.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
import copy
import sys
import time
from math import sqrt
from cmath import sqrt as sqrtc
import numpy as np

from . ktensor import K_TENSOR
from . sptensor import SP_TENSOR
from . tenmat import Tenmat
from . tensor import TENSOR


class CP_APR_MU:

    def __init__(self, epsilon=1e-10, kappa=1e-2, kappa_tol=1e-10, max_inner_iters=10,
                 n_iters=1000, print_inner_itn=0, verbose=10,
                 stoptime=1e6, tol=1e-4, random_state=42, return_type='numpy'):
        """
        Initilize the CP_APR_MU class.

        Parameters
        ----------
        epsilon : float, optional
            Prevents zero division. Default is 1e-10.
        kappa : float, optional
            Fix slackness level. Default is 1e-2.
        kappa_tol : float, optional
            Tolerance on complementary slackness. The default is 1e-10.
        max_inner_iters : int, optional
            Number of inner iterations per epoch. Default is 10.
        n_iters : int, optional
            Number of iterations during optimization or epoch. Default is 1000.
        print_inner_itn : int, optional
            Print every *n* inner iterations. Does not print if 0. Default is 0.
        verbose : int, optional
            Print every n epoch, or ``n_iters``. Does not print if 0. Default is 10.
        stoptime : float, optional
            Number of seconds before early stopping. Default is 1e6.
        tol : float, optional
            KKT violations tolerance. Default is 1e-4.
        random_state : int, optional
            Random seed for initial M.
            The default is 42.

        """

        self.epsilon = epsilon
        self.tol = tol
        self.stoptime = stoptime
        self.maxOuterIters = n_iters
        self.kappa = kappa
        self.kappaTol = kappa_tol
        self.maxInnerIters = max_inner_iters
        self.verbose = verbose
        self.printInnerItn = print_inner_itn
        self.random_state = random_state

        self.kktViolations = -np.ones(n_iters)
        self.nInnerIters = np.zeros(n_iters)
        self.times = np.zeros(n_iters)
        self.logLikelihoods = np.ones(n_iters)
        self.obj = 0

        self.M = None
        self.X = None

        self.start_time = -1
        self.exec_time = -1

        self.return_type = return_type

    def train(self, tensor=[], coords=[], values=[], rank=2, Minit='random', Type='sptensor'):
        """
        Factorize the tensor X (i.e. compute the KRUSKAL tensor M).

        Parameters
        ----------
        tensor : array
            Original dense tensor X.\n
            Use with Type = 'tensor' and pass the tensor parameter as a dense Numpy array.\n
        coords : Numpy array (i.e. array that is a list of list)
            Array of non-zero coordinates for sparse tensor X. COO format.\n
            Each entry in this array is a coordinate of a non-zero value in the tensor.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            len(Coords) is number of total entiries in X, and len(coords[0]) should give the number of dimensions.
        values : Numpy array (i.e. list of non-zero values corresponding to each list of non-zero coordinates)
            Array of non-zero tensor entries. COO format.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            Length of values must match the length of coords.
        rank : int
            Tensor rank, i.e. number of components to extract.
            The default is 2.
        Minit : string or dictionary of latent factors
            Initial value of latent factors.\n
            If Minit = 'random', initial factors are chosen randomly from uniform distribution between 0 and 1.\n
            Else, pass dictionary where the key is the mode number and value is array size d x r
            where d is the number of elements on the dimension and r is the rank.\n
            The default is "random".\n
        Type : string
            Type of tensor (i.e. sparse or dense).\n
            Use 'sptensor' for sparse, and 'tensor' for dense tensors.\n
            If 'sptensor' used, pass the list of non-zero coordinates using the Coords parameter
            and the corresponding list of non-zero elements with values parameter.\n
            The default is 'sptensor'.

        Returns
        -------
        result : dict
            KRUSKAL tensor M is returned.
            The latent factors can be found with the key 'Factors'.\n
            The weight of each component can be found with the key 'Weights'.

        """

        if rank <= 0:
            sys.exit('Number of components requested must be positive')

        # Setup for iterations
        X, M = self.__setup(tensor, coords, values, Minit, rank, Type)

        Phi = dict()
        kktModeViolations = np.zeros(X.Dimensions)
        nViolations = np.zeros(self.maxOuterIters)

        self.start_time = time.time()

        # Iterate until convergence or early stop
        for outer_iter in range(self.maxOuterIters):

            isConverged = True
            for d in range(X.Dimensions):

                # Adjust latent factors that are violating the slackness.
                if outer_iter > 1:
                    V = (Phi[str(d)] > 1) & (M.Factors[str(d)] < self.kappaTol)

                    if np.any(V.flatten('F')):
                        nViolations[outer_iter] += 1
                        M.Factors[str(d)][V > 0] += self.kappa

                # Absorb the component weight to dimension d
                M.redistribute(d)

                # Product of all matrices but the d-th
                Pi = self.__calculatePi(M, X, d)

                # Multiplicative updates
                for inner_iter in range(self.maxInnerIters):

                    self.nInnerIters[outer_iter] += 1

                    # Matrix for multiplicative update
                    Phi[str(d)] = self.__calculatePhi(M, X, d, Pi)

                    # Check for convergence
                    x = np.minimum(M.Factors[str(d)], 1 - Phi[str(d)])
                    kktModeViolations[d] = np.max(np.abs(self.__vectorizeForMu(x)))

                    if kktModeViolations[d] < self.tol:
                        break
                    else:
                        isConverged = False

                    # Do the multiplicative update
                    M.Factors[str(d)] = np.multiply(M.Factors[str(d)], Phi[str(d)])

                    # Print status
                    if self.printInnerItn != 0 and (inner_iter % self.printInnerItn == 0):
                        print("Mode = %d, Inner Iter = %d, KKT Violation = %.6f" % \
                              (d, inner_iter + 1, kktModeViolations[d]))

                M = M.normalize(M, mode=d)

            self.kktViolations[outer_iter] = np.max(kktModeViolations)

            # calculate the log likelihood
            M_ = M.normalize(copy.deepcopy(M), N=-2)
            obj_ = self.__tt_loglikelihood(M_, X)
            self.logLikelihoods[outer_iter] = obj_

            # Print update
            if self.verbose != 0 and (outer_iter % self.verbose == 0):
                print("Iter=%d, Inner Iter=%d, KKT Violation=%.6f, obj=%.6f, nViolations=%d" % \
                      (outer_iter + 1, self.nInnerIters[outer_iter], self.kktViolations[outer_iter], \
                       self.logLikelihoods[outer_iter], nViolations[outer_iter]))

            # Check for convergence
            if isConverged:
                if self.verbose != 0:
                    print("Exiting because all subproblems reached KKT tol.")
                break

            self.times[outer_iter] = time.time() - self.start_time
            if self.times[-1] > self.stoptime:
                if self.verbose != 0:
                    print("Exiting because time limit exceeded.")
                break

        # Done
        result = self.__finalize(M, X, outer_iter)

        return result

    def __finalize(self, M, X, outer_iter):
        """
        Helper functions to finalize the results. Calculates the final fit, performs the final
        tensor format conversions, shows the results if verbose, and returns the
        KRUSKAL tensor M.

        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor_Torch.K_TENSOR
        X : class
            Original tensor. sptensor_Torch.SP_TENSOR
        outer_iter : int
            Current epoch.

        Returns
        -------
        result : dict
            KRUSKAL tensor M is returned.
            The latent factors can be found with the key 'Factors'.\n
            The weight of each component can be found with the key 'Weights'.

        """

        # Clean up final result
        M = M.normalize(M, N=-2)
        self.obj = self.__tt_loglikelihood(copy.deepcopy(M), X)

        self.M = M
        self.X = X

        self.exec_time = time.time() - self.start_time

        if self.verbose != 0:

            if X.Type == 'sptensor':
                normX = np.linalg.norm(X.Values)
            elif X.Type == 'tensor':
                normX = np.linalg.norm(X.Tensor.flatten())

            nrm_sqr = M.norm() ** 2
            rem = M.innerprod(X)

            try:
                normresidual = sqrt(normX ** 2 + nrm_sqr - 2 * rem)
            # if negative in sqrt
            except Exception as e:
                normresidual = sqrtc(normX ** 2 + nrm_sqr - 2 * rem)

            fit = 1 - (normresidual / normX)

            print("===========================================")
            print(" Final log-likelihood = %f" % self.obj)
            print(" Final least squares fit = %f" % fit)
            print(" Final KKT violation = %f" % self.kktViolations[outer_iter])
            print(" Total inner iterations = %d" % np.sum(self.nInnerIters))
            print(" Total execution time = %.4f seconds" % self.exec_time)

        result = dict()
        if self.return_type == 'numpy':
            if self.verbose != 0:
                print("Converting the latent factors to Numpy arrays.")

            if M.Rank == 1:
                for dim in range(M.Dimensions):
                    M.Factors[str(dim)] = [item for sublist in M.Factors[str(dim)] for item in sublist]
                    M.Factors[str(dim)] = np.array(M.Factors[str(dim)])

                result['Factors'] = M.Factors

            else:
                for dim in range(M.Dimensions):
                    M.Factors[str(dim)] = np.array(M.Factors[str(dim)])
                result['Factors'] = M.Factors
            result['Weights'] = np.array(M.Weights)

        else:
            result['Factors'] = M.Factors
            result['Weights'] = M.Weights

        return result

    def __setup(self, Tensor, Coords, Values, Minit, Rank, Type):
        """


        Parameters
        ----------
        Tensor : array
            Original dense tensor X.\n
            Use with Type = 'tensor' and pass the tensor parameter as a dense Numpy array.\n
        Coords : Numpy array (i.e. array that is a list of list)
            Array of non-zero coordinates for sparse tensor X. COO format.\n
            Each entry in this array is a coordinate of a non-zero value in the tensor.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            len(Coords) is number of total entiries in X, and len(coords[0]) should give the number of dimensions.
        Values : Numpy array (i.e. list of non-zero values corresponding to each list of non-zero coordinates)
            Array of non-zero tensor entries. COO format.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            Length of values must match the length of coords.
        Rank : int
            Tensor rank, i.e. number of components to extract.
            The default is 2.
        Minit : string or dictionary of latent factors
            Initial value of latent factors.\n
            If Minit = 'random', initial factors are chosen randomly from uniform distribution between 0 and 1.\n
            Else, pass dictionary where the key is the mode number and value is array size d x r
            where d is the number of elements on the dimension and r is the rank.\n
            The default is "random".\n
        Type : string
            Type of tensor (i.e. sparse or dense).\n
            Use 'sptensor' for sparse, and 'tensor' for dense tensors.\n
            If 'sptensor' used, pass the list of non-zero coordinates using the Coords parameter
            and the corresponding list of non-zero elements with values parameter.\n
            The default is 'sptensor'.

        Returns
        -------
        M : class
            KRUSKAL tensor M class. ktensor.K_TENSOR
        X : class
            Original tensor. sptensor.SP_TENSOR or tensor.TENSOR

        """

        # Sparse tensor
        if Type == 'sptensor':
            if len(Coords) == 0:
                sys.exit('Coordinates of the non-zero elements is not passed for sptensor.\
                         Use the Coords parameter.')
            if len(Values) == 0:
                sys.exit('Non-zero values are not passed for sptensor.\
                         Use the Values parameter')
            if (Coords < 0).all():
                sys.exit('Data tensor must be nonnegative for Poisson-based factorization')

            X = SP_TENSOR(Coords, Values)

        # Dense tensor
        elif Type == 'tensor':

            if len(Tensor) == 0:
                sys.exit('Tensor is not passed for dense tensor type.\
                         Use the Tensor parameter.')
            if (Tensor < 0).all():
                sys.exit('Data tensor must be nonnegative for Poisson-based factorization')

            X = TENSOR(Tensor)

        M = K_TENSOR(Rank, X.Size, Minit, self.random_state)

        M = M.normalize(M)

        if self.verbose != 0:
            print("CP-APR (MU):")

        return X, M

    def __tt_loglikelihood(self, M, X):
        """
        This function computes log-likelihood of tensor X with model M.

        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor.K_TENSOR
        X : class
            Original tensor. sptensor.SP_TENSOR or tensor.TENSOR

        Returns
        -------
        f : float
            log-likelihood.

        """

        M = M.normalize(M, N=1)
        f = 0

        if X.Type == 'sptensor':

            A = M.Factors[str(0)][X.Coords[:, 0], :]

            for d in range(1, X.Dimensions):
                A = np.multiply(A, M.Factors[str(d)][X.Coords[:, d], :])

            f = np.sum(np.multiply(X.Values, np.log(np.sum(A, 1)))) - \
                np.sum(np.sum(M.Factors[str(0)], axis=0))

        elif X.Type == 'tensor':

            dX = Tenmat(X, 0).Tensor
            dM = Tenmat(M, 0).Tensor

            for ii in range(dX.shape[0]):
                for jj in range(dM.shape[1]):
                    if dX[ii, jj] == 0.0:
                        pass
                    else:
                        f += dX[ii, jj] * np.log(dM[ii, jj])

            f -= np.sum(np.sum(M.Factors['0']))

        return f

    def __vectorizeForMu(self, x):
        """
        Turn x into a vector.

        Parameters
        ----------
        x : array
            tensor x.

        Returns
        -------
        y : array
            flattenned x.

        """

        y = x.flatten('F')
        return y

    def __calculatePhi(self, M, X, mode, Pi):
        """
        This function calculates the matrix for MU

        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor.K_TENSOR
        X : class
            Original tensor. sptensor.SP_TENSOR or tensor.TENSOR
        mode : int
            dimension.
        Pi : array
            product of all matrices but nth.

        Returns
        -------
        Phi : array
            multiplicative update.

        """

        Phi = 0

        if X.Type == 'sptensor':

            Phi = -np.ones((X.Size[mode], M.Rank))
            xsubs = X.Coords[:, mode]

            # not sure about the one below
            v = np.sum(np.multiply(M.Factors[str(mode)][xsubs, :], Pi), 1)
            wvals = np.divide(X.Values, np.maximum(v, self.epsilon))

            for r in range(M.Rank):
                Yr = np.bincount(xsubs, np.multiply(wvals, Pi[:, r]), X.Size[mode])
                Phi[:, r] = Yr

        elif X.Type == 'tensor':

            Xn = Tenmat(X, mode).Tensor
            V = np.dot(M.Factors[str(mode)], Pi.T)
            W = np.divide(Xn, np.maximum(V, self.epsilon))
            Y = np.dot(W, Pi)
            Phi = Y

        return Phi

    def __calculatePi(self, M, X, mode):
        """
        Calculate product of all matrices but the n-th.

        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor.K_TENSOR
        X : class
            Original tensor. sptensor.SP_TENSOR or tensor.TENSOR
        mode : int
            dimension.

        Returns
        -------
        Pi : array
            product of all matrices but nth..

        """

        Pi = 0

        if X.Type == 'sptensor':

            Pi = np.ones((len(X.Coords), M.Rank))

            for nn in range(X.Dimensions):
                if nn != mode:
                    Pi = np.multiply(M.Factors[str(nn)][X.Coords[:, nn], :], Pi)

        elif X.Type == 'tensor':

            nn = [str(x) for x in range(0, X.Dimensions) if x != mode]
            Pi = M.khatrirao(nn)

        return Pi
