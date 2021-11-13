#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of the CP-APR algorithm [1-4] with PyTorch backend.\n
This backend can be used to factorize sparse tensorsin COO format on GPU or CPU.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
import copy
import time
from math import sqrt
from cmath import sqrt as sqrtc
from tqdm import tqdm

import numpy as np
import torch as tr

from . ktensor_Torch import K_TENSOR
from . sptensor_Torch import SP_TENSOR

from . redistribute_ktensor import redistribute
from . normalize_ktensor import normalize
from . innerprod_ktensor import innerprod
from . norm_ktensor import norm

class CP_APR_MU:

    def __init__(self, epsilon=1e-10, kappa=1e-2, kappa_tol=1e-10, max_inner_iters=10,
                 n_iters=1000, print_inner_itn=0, verbose=10, simple_verbose=False,
                 stoptime=1e6, tol=1e-4, random_state=42, device='cpu',
                 device_num='0', return_type='numpy', dtype='torch.DoubleTensor',
                 follow_M=False):
        """
        Initilize the CP_APR_MU class. Sets up the class variables and the CUDA for
        tensors.

        Parameters
        ----------
        epsilon : float, optional
            Prevents zero division. Default is 1e-10.
        kappa : float, optional
            Fix slackness level. Default is 1e-2.
        kappa_tol : float, optional
            Tolerance on slackness level. Default is 1e-10.
        max_inner_iters : int, optional
            Number of inner iterations per epoch. Default is 10.
        n_iters : int, optional
            Number of iterations during optimization or epoch. Default is 1000.
        print_inner_itn : int, optional
            Print every *n* inner iterations. Does not print if 0. Default is 0.
        verbose : int, optional
            Print every n epoch, or ``n_iters``. Does not print if 0. Default is 10.
        simple_verbose : bool, optional
            Turns off details for verbose, such as fit, but instead shows a progress bar.
        stoptime : float, optional
            Number of seconds before early stopping. Default is 1e6.
        tol : float, optional
            KKT violations tolerance. Default is 1e-4.
        random_state : int, optional
            Random seed for initial M.
            The default is 42.
        device : string, optional
            Torch device to be used.
            'cpu' to use PyTorch with CPU.
            'gpu' to use cuda:0
            The default is cpu.
        device_num : string, optional
            Which device to to store the tensors.
        return_type : string, optional
            Type for the latent factors.
            'torch' keep as torch tensors.
            'numpy' convert to numpy arrays.
        dtype : string, optional
            Type to be used in torch tensors.
            Default is torch.cuda.DoubleTensor.
        follow_M : bool, optional
            Saves M on each iteration.
            The default is False.

        """
        # Parameter for printing
        self.verbose = verbose
        self.simple_verbose = simple_verbose
        self.print_inner_itn = print_inner_itn

        # Keep track of the runtime and the iteration stoptime
        self.start_time = -1
        self.final_iter = -1
        
        # Keep track of Ms
        self.follow_M = follow_M
        self.saved_Ms = list()

        # Set the default tensor type
        if device == 'gpu' and dtype == 'torch.DoubleTensor':
            dtype = 'torch.cuda.DoubleTensor'

        self.dtype = dtype
        tr.set_default_tensor_type(self.dtype)

        # GPU or CPU device parameters
        self.device = device
        self.device_num = str(device_num)

        if device == 'gpu':
            if tr.cuda.is_available():
                self.device = tr.device('cuda:' + self.device_num)
                if self.verbose != 0:
                    print('Using', tr.cuda.get_device_name(int(self.device_num)))
            else:
                raise Exception('No CUDA device found')

        # Return Format
        if return_type in ['torch', 'numpy']:
            self.return_type = return_type
        else:
            raise Exception('Invalid return type!')

        # Original X tensor, and KRUSKAL tensor M
        self.X = None
        self.M = None

        # Optimization Parameters
        self.tol = tol
        self.stoptime = stoptime
        self.exec_time = -1
        self.n_iters = n_iters
        self.max_inner_iters = max_inner_iters
        self.random_state = random_state
        self.kappa = kappa
        self.kappa_tol = kappa_tol

        self.kktViolations = -tr.ones(n_iters).to(self.device)
        self.nInnerIters = tr.zeros(n_iters).to(self.device)
        self.times = tr.zeros(n_iters).to(self.device)
        self.logLikelihoods = tr.ones(n_iters).to(self.device)
        self.epsilon = tr.tensor(epsilon).to(self.device)
        self.obj = 0


    def train(self, tensor=[], coords=[], values=[], rank=2, Minit='random', Type='sptensor'):
        """
        Factorize the tensor X (i.e. compute the KRUSKAL tensor M).

        Parameters
        ----------
        tensor : PyTorch Sparse Tensor or dense Numpy array as tensor
            Original dense or sparse tensor X.\n
            Can be used when Type = 'sptensor'. Then tensor parameter needs to be a PyTorch Sparse tensor.\n
            Or use with Type = 'tensor' and pass the tensor parameter as a dense Numpy array.\n
            Note that PyTorch only supports Type = 'sptensor'.
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
            'sptensor' can be used with method = 'torch', method = 'numpy'.\n
            If 'sptensor' used, pass the list of non-zero coordinates using the Coords parameter
            and the corresponding list of non-zero elements with values.\n
            'sptensor' can also be used with the PyTorch Sparse format. Pass the torch.sparse format in the tensor parameter.\n
            'tensor' can be used with method = 'numpy' only. Pass the tensor using tensor parameter in that case.\n
            The default is 'sptensor'.

        Returns
        -------
        result : dict
            KRUSKAL tensor M is returned.
            The latent factors can be found with the key 'Factors'.\n
            The weight of each component can be found with the key 'Weights'.

        """

        if rank <= 0:
            raise Exception('Number of components requested must be positive!')

        # Setup for iterations
        X, M = self.__setup(tensor, coords, values, Minit, rank, Type)

        Phi = dict()
        kktModeViolations = tr.zeros(X.Dimensions)
        nViolations = tr.zeros(self.n_iters)

        self.start_time = time.time()

        # Iterate until convergence or early stop
        for outer_iter in tqdm(range(self.n_iters), disable=not(self.simple_verbose)):

            isConverged = True
            for d in range(X.Dimensions):

                # Adjust latent factors that are violating the slackness.
                if outer_iter > 1:
                    V = (Phi[str(d)] > 1) & (M.Factors[str(d)] < self.kappa_tol)

                    if tr.any(V.flatten().transpose(-1, 0).flatten()):
                        nViolations[outer_iter] += 1
                        M.Factors[str(d)][V > 0] += self.kappa

                # Absorb the component weight to dimension d
                M = redistribute(M, d)
                
                # Product of all matrices but the d-th
                Pi = self.__calculatePi(M, X, d)

                # Multiplicative updates
                for inner_iter in range(self.max_inner_iters):

                    self.nInnerIters[outer_iter] += 1

                    # Matrix for multiplicative update
                    Phi[str(d)] = self.__calculatePhi(M, X, d, Pi)

                    # Check for convergence
                    x = tr.min(M.Factors[str(d)], 1 - Phi[str(d)])
                    kktModeViolations[d] = tr.max(tr.abs(self.__vectorizeForMu(x)))

                    if kktModeViolations[d] < self.tol:
                        break
                    else:
                        isConverged = False

                    # Do the multiplicative update
                    M.Factors[str(d)] = tr.mul(M.Factors[str(d)], Phi[str(d)])

                    # Print status
                    if self.print_inner_itn != 0 and (inner_iter % self.print_inner_itn == 0) and (self.simple_verbose == False):
                        print("Mode = %d, Inner Iter = %d, KKT Violation = %.6f" % \
                              (d, inner_iter + 1, kktModeViolations[d]))

                M = normalize(M, mode=d)

            self.kktViolations[outer_iter] = tr.max(kktModeViolations)

            # calculate the log likelihood
            M_ = normalize(copy.deepcopy(M), N=-2)
            obj_ = self.__tt_loglikelihood(M_, X)
            self.logLikelihoods[outer_iter] = obj_
            
            # if we want to save the results from current iteration
            if self.follow_M:
                save_M = dict()
                if self.return_type == 'numpy':
                    save_M = result = self.__transfer_M_cpu(M_)
                else:
                    save_M['Factors'] = M_.Factors
                    save_M['Weights'] = M_.Weights
                self.saved_Ms.append(save_M)
            

            # Print update
            if self.verbose != 0 and (outer_iter % self.verbose == 0) and (self.simple_verbose == False):
                print("Iter=%d, Inner Iter=%d, KKT Violation=%.6f, obj=%.6f, nViolations=%d" % \
                      (outer_iter + 1, self.nInnerIters[outer_iter], self.kktViolations[outer_iter], \
                       self.logLikelihoods[outer_iter], nViolations[outer_iter]))

            # Check for convergence
            if isConverged:
                if self.verbose != 0 and (self.simple_verbose == False):
                    print("Exiting because all subproblems reached KKT tol.")
                break

            self.times[outer_iter] = time.time() - self.start_time
            if self.times[-1] > self.stoptime:
                if self.verbose != 0 and (self.simple_verbose == False):
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
        M = normalize(M, N=-2)
        self.obj = self.__tt_loglikelihood(copy.deepcopy(M), X)
        self.final_iter = outer_iter + 1

        result = dict()
        self.exec_time = time.time() - self.start_time

        if self.verbose != 0 and (self.simple_verbose == False):
            normX = tr.norm(X.data)
            nrm_sqr = norm(M) ** 2
            rem = innerprod(M, X)

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
            print(" Total inner iterations = %d" % tr.sum(self.nInnerIters))
            print(" Total execution time = %.4f seconds" % self.exec_time)

        if self.return_type == 'numpy':
            if self.verbose != 0 and (self.simple_verbose == False):
                print("Converting the latent factors to Numpy arrays.")

            # Convert KTENSOR to Numpy arrays
            result = self.__transfer_M_cpu(M)

            # convert the optimization variables
            self.kktViolations = self.kktViolations.cpu().numpy().astype('float64')
            self.nInnerIters = self.nInnerIters.cpu().numpy().astype('float64')
            self.times = self.times.cpu().numpy().astype('float64')
            self.logLikelihoods = self.logLikelihoods.cpu().numpy().astype('float64')
            self.epsilon = self.epsilon.cpu().numpy().astype('float64')
            self.obj = float(self.obj.cpu().numpy().astype('float64') )

        else:
            result['Factors'] = M.Factors
            result['Weights'] = M.Weights

        # if GPU used, free up the space
        if not isinstance(self.device, str) and self.device.type == 'torch':
            tr.cuda.empty_cache()

        # Save the KRUSKAL tensor M and the original tensor X
        self.M = M
        self.X = X

        return result
    
    def __transfer_M_cpu(self, M):
        """
        Transfers M to CPU if requested.
        
        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor_Torch.K_TENSOR
        
        Returns
        -------
        M : dict
            M factors and weight that is transferred to CPU.
        
        """
        result = dict()
        if M.Rank == 1:
            for dim in range(M.Dimensions):
                M.Factors[str(dim)] = M.Factors[str(dim)].cpu().numpy().astype('float64')
                M.Factors[str(dim)] = [item for sublist in M.Factors[str(dim)] for item in sublist]
                M.Factors[str(dim)] = np.array(M.Factors[str(dim)])

        else:
            for dim in range(M.Dimensions):
                M.Factors[str(dim)] = M.Factors[str(dim)].cpu().numpy().astype('float64')
                    
        result['Factors'] = M.Factors
        result['Weights'] = M.Weights.cpu().numpy().astype('float64')
            
        return result

    def __setup(self, Tensor, Coords, Values, Minit, Rank, Type):
        """
        Sets up the classes for KRUSKAL tensor and the original tensor X.

        Parameters
        ----------
        Tensor : PyTorch Sparse Tensor or dense Numpy array as tensor
            Original dense or sparse tensor X.\n
            Can be used when Type = 'sptensor'. Then Tensor needs to be a PyTorch Sparse tensor.\n
            Or use with Type = 'tensor' and pass Tensor as a dense Numpy array.\n
            Note that PyTorch only supports Type = 'sptensor'.
        Coords : Numpy array (i.e. array that is a list of list)
            Array of non-zero coordinates for sparse tensor X. COO format.\n
            Each entry in this array is a coordinate of a non-zero value in the tensor.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            len(Coords) is number of total entiries in X, and len(coords[0]) should give the number of dimensions.
        Values : Numpy array (i.e. list of non-zero values corresponding to each list of non-zero coordinates)
            Array of non-zero tensor entries. COO format.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            Length of values must match the length of coords.
        Rank : int or list
            Tensor rank, or list of ranks for two tensors.\n
            List of ranks will allow using weighted prediction between the two latent factors.\n
            Pass a single integer or list of length two.\n
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
            'sptensor' can be used with method = 'torch', method = 'numpy'.\n
            If 'sptensor' used, pass the list of non-zero coordinates usingvCoords parameter
            and the corresponding list of non-zero elements with values.\n
            'sptensor' can also be used with the PyTorch Sparse format. Pass the torch.sparse format in the tensor parameter.\n
            'tensor' can be used with method = 'numpy'. Pass the tensor using tensor parameter in that case.\n
            The default is 'sptensor'.

        Returns
        -------
        M : class
            KRUSKAL tensor M class. ktensor_Torch.K_TENSOR
        X : class
            Original tensor. sptensor_Torch.SP_TENSOR

        """

        # Setup the optimization variables
        self.kktViolations = -tr.ones(self.n_iters).to(self.device)
        self.nInnerIters = tr.zeros(self.n_iters).to(self.device)
        self.times = tr.zeros(self.n_iters).to(self.device)
        self.logLikelihoods = tr.ones(self.n_iters).to(self.device)
        self.epsilon = tr.tensor(self.epsilon).to(self.device)
        self.obj = 0

        # Setup the tensors
        if Type == 'sptensor':
            if tr.is_tensor(Tensor):
                if Tensor._nnz() == 0:
                    raise Exception('Non-zero values must be more than 0.')

            else:
                if len(Coords) == 0:
                    raise Exception('Coordinates of the non-zero elements is not passed for sptensor.\
                             Use the Coords parameter.')
                if len(Values) == 0:
                    raise Exception('Non-zero values are not passed for sptensor.\
                             Use the Values parameter')
                if (Coords < 0).all():
                    raise Exception('Coords tensor must be nonnegative for factorization')

            # Convert the initial latent factors to pyTorch tensors
            if Minit != 'random':
                flag = False
                if "Factors" in Minit and isinstance(Minit["Factors"]['0'], (list, np.ndarray)):
                    flag=True
                elif isinstance(Minit['0'], (list, np.ndarray)):
                    flag=True
                
                if flag:
                    if "Factors" in Minit:
                        for d in range(len(Minit["Factors"].keys())):
                            Minit["Factors"][str(d)] = tr.from_numpy(Minit["Factors"][str(d)]).type(self.dtype)
                            
                    else:
                        comp = len(Minit.keys())
                        if "Weights" in Minit:
                            comp -= 1
                        
                        for d in range(comp):
                            Minit[str(d)] = tr.from_numpy(Minit[str(d)]).type(self.dtype)
                    
            X = SP_TENSOR(Tensor, Coords, Values, self.dtype, self.device)


        elif Type == 'tensor':
            raise Exception("PyTorch backend only support sparse tensor implementation currently.")

        M = K_TENSOR(Rank, X.Size, Minit, self.random_state, self.device, self.dtype)
        M = normalize(M)
        
        if self.verbose != 0 and (self.simple_verbose == False):
            print("CP-APR (MU):")

        return X, M

    def __tt_loglikelihood(self, M, X):
        """
        This function computes log-likelihood of tensor X with model M.

        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor_Torch.K_TENSOR
        X : class
            Original tensor. sptensor_Torch.SP_TENSOR

        Returns
        -------
        f : float
            log-likelihood.

        """

        M = normalize(M, N=1)

        A = M.Factors[str(0)][X.Coords[:, 0], :]

        for d in range(1, X.Dimensions):
            A = tr.mul(A, M.Factors[str(d)][X.Coords[:, d], :])

        f = tr.sum(tr.mul(X.data, tr.log(tr.sum(A, 1)))) - \
            tr.sum(tr.sum(M.Factors[str(0)], axis=0))

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

        y = x.flatten().transpose(-1, 0).flatten()
        return y

    def __calculatePhi(self, M, X, mode, Pi):
        """
        Calculates the matrix for MU

        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor_Torch.K_TENSOR
        X : class
            Original tensor. sptensor_Torch.SP_TENSOR
        mode : int
            dimension.
        Pi : array
            Product of all matrices but nth.

        Returns
        -------
        Phi : array
            multiplicative update.

        """

        Phi = -tr.ones((X.Size[mode], M.Rank)).to(self.device)
        xsubs = X.Coords[:, mode]

        v = tr.sum(tr.mul(M.Factors[str(mode)][xsubs, :], Pi), 1)
        wvals = tr.div(X.data, tr.max(v, self.epsilon))

        for r in range(M.Rank):
            Yr = tr.bincount(xsubs, tr.mul(wvals, Pi[:, r]), X.Size[mode])
            Phi[:, r] = Yr

        return Phi

    def __calculatePi(self, M, X, mode):
        """
        Calculates the product of all matrices without the "mode" dimension.

        Parameters
        ----------
        M : class
            KRUSKAL tensor M class. ktensor_Torch.K_TENSOR
        X : class
            Original tensor. sptensor_Torch.SP_TENSOR
        mode : int
            dimension.

        Returns
        -------
        Pi : array
            product of all matrices but nth.

        """

        Pi = tr.ones((X.nnz, M.Rank)).to(self.device)

        for nn in range(X.Dimensions):
            if nn != mode:
                Pi = tr.mul(M.Factors[str(nn)][X.Coords[:, nn], :], Pi)

        return Pi
