#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_anomaly_detection.py performs p-value scoring over the tensor decomposition, i.e.
the KRUSKAL tensor M. The calculated p-values are used to detect anomalies.\n
This method was introduced by Eren et al. in [1].

CyberToaster, Project 1, Summer 2020\n
Los Alamos National Laboratory\n

Anomaly detection using Tensors and their Decompositions.\n
Student: Maksim E. Eren\n
Primary Mentor: Juston Moore\n
Secondary Mentors: Boian Alexandrov and Patrick Avery

References
========================================
[1] M. E. Eren, J. S. Moore and B. S. Alexandro, "Multi-Dimensional Anomalous Entity Detection via Poisson Tensor Factorization," 2020 IEEE International Conference on Intelligence and Security Informatics (ISI), 2020, pp. 1-6, doi: 10.1109/ISI49825.2020.9280524.

@author: Maksim Ekin Eren, Juston S. Moore
"""
import sys
import numpy as np
import scipy.special as sc
from scipy.special import chdtrc as chi2_cdf
from scipy.stats import poisson


class PoissonTensorAnomaly():
    """
    Anomaly detection using Poisson Distribution and Canonical Polyadic (CP)
    with Alternating Poisson Regression tensor decomposition (CP-APR).

    Componenets of the CP-APR used to calculate the p-values for each instance
    through Poisson cumulative distribution function (cdf).

    p-values are then used to determine if the event is an anomaly.
    Lower p-values are more anomalous.

    v2: Utilizes Numpy vectorization for the calculations.


    References:\n
    1) Chi, Eric C. and Tamara G. Kolda. “On Tensors, Sparsity, and Nonnegative
    Factorizations.” SIAM J. Matrix Anal. Appl. 33 (2012): 1272-1299.\n
    2) Turcotte, Melissa J. M. et al. “Unified Host and Network Data Set.
    ” ArXiv abs/1708.07518 (2017): n. pag.\n
    3) Wikipedia contributors. "Poisson distribution." Wikipedia,
    The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 29 Jun. 2020. Web. 6 Jul. 2020.
    """

    # LIBRARY INFORMATION
    __version_info__ = ('2', '1', '0')
    __version__ = '.'.join(__version_info__)
    __author__ = "Maksim E. Eren, Juston Moore"
    __info__ = "Anomaly detection using Poisson Distribution and Canonical Polyadic (CP) \
    with Alternating Poisson Regression tensor decomposition (CP-APR). This version utilize \
    Numpy vectorization in the calculations."

    def __init__(self, dimensions=dict(), weights=list(),
                 objective='p_value', lambda_method='single_tensor',
                 p_value_fusion_index=[0],
                 ensemble_dimensions=dict(), ensemble_weights=list(), ensemble_significance=[0.1, 0.9],
                 mode_weights=[1], ignore_dimensions_indx=[]):
        r"""
        Initilize the anomaly detector class.

        Parameters
        ----------
        dimensions : dict, required
            Components of the KRUSKAL Tensor Decomposition. The default is dict.\n
            Each element is a dimension (factors of a component) and
            each dimension has (nxK) elements for that factor for rank K.
        weights : list, required
            Weights of each component of parameter dimensions. The default is list.
        objective : string, optional
            What to calculate.\n
            Options: p_value, p_value_fusion_harmonic, p_value_fusion_harmonic_observed,
            p_value_fusion_chi2, p_value_fusion_chi2_observed, p_value_fusion_arithmetic,
            log_likelihood
            The default is 'p_value'.
        lambda_method : string, optional
            How to calculate lambda.\n
            If 'single_tensor', it will use single ktensor passed in dimensions
            when calculating lambda.\n
            If 'ensemble', it will use two ktensors where parameter dimensions
            is a K>=1 rank tensor with lambda weight ensemble_significance[0] and
            parameter ensemble_dimensions is a ktensor with K>1 rank tensor with
            lambda weight ensemble_significance[1]. The default is 'single_tensor'.
        p_value_fusion_index :  list
            Index to fix, or calculate the p-value fusions. Only used when
            objective is set to p_value_fusion.
            The default is [0].
        ensemble_dimensions : dict, optional
            Components of the KRUSKAL Tensor Decomposition.\n
            Each element is a dimension (factors of a component) and
            each dimension has (nxK) elements for that factor for rank K.\n
            This is the second ktensor dimension passed. It will be used if
            lambda_method is set to 'ensemble'.
            Its lambda weight is ensemble_significance[1]. The default is dict().
        ensemble_weights : list, optional
            Weights of each component of ensemble_dimensions. The default is list().
            Only used if lambda_method is 'ensemble'.
        ensemble_significance : list, optional
            lambda weight of each ktensor when using 'ensemble' lambda_method.\n
            Weight of dimensions: ensemble_significance[0]\n.
            Weight of ensemble_dimensions: ensemble_significance[1]\n
            The default is [0.1, 0.9].
        mode_weights : list, optional
            Weight of each dimension.\n
            The default is [1].
        ignore_dimensions_indx : list, optional
            If any dimension in latent factors should be ignored when calculating the lambdas.\n
            The default is [].


        """

        # Store KRUSKAL Tensor componenets
        # Each index in dimensions is a dimension from the tensor
        # and each dimension consist of that dimension's values and Ranks
        self.dimensions = list()
        for _, value in dimensions.items():
            numpy_array = np.array([np.array(xi) for xi in value])
            self.dimensions.append(numpy_array)

        # K, Rank of the tensor
        # if K = 1
        if len(self.dimensions[0].shape) == 1:

            # Component weights
            self.weights = np.array([weights])
            # Rank
            self.K = 1
        else:

            # Component weights
            self.weights = np.array(weights)
            # Rank
            self.K = self.dimensions[0].shape[1]

        # D, Order or mode of the tensor. Number of dimensions
        # There are D factors within each K components
        self.D = len(self.dimensions)

        # Tensor/Dimensions Size
        self.dimensions_size = list()
        for dimension in self.dimensions:
            self.dimensions_size.append(dimension.shape[0])

        # How to calculate the scores
        avail_objectives = ['p_value', \
                            'p_value_fusion_harmonic', 'p_value_fusion_harmonic_observed', \
                            'p_value_fusion_arithmetic', \
                            'p_value_fusion_chi2', 'p_value_fusion_chi2_observed', \
                            'log_likelihood']
        if objective not in avail_objectives:
            sys.exit('Invalid objective. Available methods: ' + ','.join(avail_objectives))
        self.objective = objective

        # Lamda calculation method
        avail_lambda_methods = ['single_tensor', 'ensemble']
        if objective not in avail_objectives:
            sys.exit('Invalid lambda method. Available methods: ' + ','.join(avail_lambda_methods))
        self.lambda_method = lambda_method

        # ensemble lambda will have multiple tensors with different ranks to be used
        # when calculating the lambda
        self.ensemble_dimensions = list()
        self.ensemble_weights = list()
        self.ensemble_significance = ensemble_significance
        self.K_ensemble = -1

        # extract dimensions from each tensor in the bag
        if self.lambda_method == 'ensemble':

            for _, value in ensemble_dimensions.items():
                numpy_array = np.array([np.array(xi) for xi in value])
                self.ensemble_dimensions.append(numpy_array)

            if len(self.ensemble_dimensions[0].shape) == 1:

                self.ensemble_weights = np.array([ensemble_weights])
                self.K_ensemble = 1

            else:
                self.ensemble_weights = np.array(ensemble_weights)
                self.K_ensemble = self.ensemble_dimensions[0].shape[1]

        self.p_value_fusion_index = p_value_fusion_index

        # Weight of each dimension
        if len(mode_weights) > 1:
            if len(mode_weights) > len(self.dimensions):
                sys.exit('Mode weights must be same number of dimensions.')

            self.mode_weights = mode_weights
        else:
            self.mode_weights = mode_weights * len(self.dimensions)

        # Dimensions to ignore when calculating the lambdas
        self.ignore_dimensions_indx = ignore_dimensions_indx


    def predict(self, coords, values, from_matlab=False):
        '''
        Get the scores using the KRUSKAL components given the non-zero coordinates
        and values and the objective.

        Parameters
        ----------
        coords : list of list
            Coordinates of the non-zero elements within the sparse tensor.
        values : list
            Non-zero values that are in the sparse tensor.
        from_matlab : bool
            Set True if need to substract 1 to the coordinates, since matlab starts at 1.\n
            The default is False.

        Returns
        -------
        prediction : dict
            Dictionary of calculated objective.

        '''

        # change coords to a numpy array or arrays
        coords = np.array([np.array(xi) for xi in coords])
        # change values to a numpy array
        values = np.array(values)

        # substract 1 from each element since for Matlab +1 was done
        if from_matlab:
            coords -= 1

        # calculate Poisson Lambda values for each instance, or event
        lambdas = self.__get_lambdas(coords)

        # calculate the p_values
        if self.objective == 'p_value':
            scores = self.__get_p_values(lambdas, values)
        # calculate p_value fusion with harmonic mean
        elif self.objective == 'p_value_fusion_harmonic':
            p_values = self.__get_p_values(lambdas, values)
            scores = self.__get_p_value_fusion_harmonic(p_values, coords)
        # calculate p_value fusion with harmonic mean for observed events
        elif self.objective == 'p_value_fusion_harmonic_observed':
            p_values = self.__get_p_values(lambdas, values)
            scores = self.__get_p_value_fusion_harmonic_observed(p_values, coords)
        # calculate p_value fusion with arithmetic mean
        elif self.objective == 'p_value_fusion_arithmetic':
            p_values = self.__get_p_values(lambdas, values)
            scores = self.__get_p_value_fusion_arithmetic(p_values, coords)
        #  cakculate p_value fusion with chi-squared test
        elif self.objective == 'p_value_fusion_chi2':
            p_values = self.__get_p_values(lambdas, values)
            scores = self.__get_p_value_fusion_chi2(p_values, coords)
        # cakculate p_value fusion with chi-squared test for observed events
        elif self.objective == 'p_value_fusion_chi2_observed':
            p_values = self.__get_p_values(lambdas, values)
            scores = self.__get_p_value_fusion_chi2_observed(p_values, coords)
        # calculate log_likelihood
        elif self.objective == 'log_likelihood':
            scores = self.__get_log_likelihood(lambdas, coords, values)

            # prepare the return value
        prediction = dict()
        prediction[self.objective] = scores
        prediction['lambda'] = lambdas

        return prediction

    def __get_log_likelihood(self, lambdas, coords, values):
        '''
        Calculate the log-likelihood given the Poisson lambda values, coordinates
        of the non-zero elements, and list of non-zero elements.

        Parameters
        ----------
        lambdas : numpy array
            array of Poisson lambdas.
        coords : numpy array
            Coordinates of the non-zero elements within the sparse tensor.
        values : numpy array
            non-zero values of a tensor.

        Returns
        -------
        log_likelihood : float
            log-likelihood score of given coordinates.

        '''

        # Calculate the non-zero log sum
        log_sum_nnz = 0
        for index, lambd in enumerate(lambdas):
            log_sum_nnz += ((values[index] * np.log(lambd)) - sc.gammaln(values[index] + 1))

        # Calculate Lambda (capital lambda)
        theta = 1
        for dimension_index, dimension in enumerate(self.dimensions):
            if self.K == 1:  # if rank is 1, no components to sum
                theta *= (dimension[coords[:, dimension_index]] * self.weights)
            else:
                theta *= (dimension[coords[:, dimension_index]] * self.weights).sum(axis=1)

        Lambda = theta.sum()
        if self.lambda_method == 'ensemble':
            # main ktensor's lambdas weight
            Lambda *= self.ensemble_significance[0]

            # Calculate Lambda_2 (capital lambda 2), Lambda for the secondary tensor
            theta = 1
            for dimension_index, dimension in enumerate(self.ensemble_dimensions):
                if self.K_ensemble == 1:  # if rank is 1, no components to sum
                    theta *= (dimension[coords[:, dimension_index]] * self.ensemble_weights)
                else:
                    theta *= (dimension[coords[:, dimension_index]] * self.ensemble_weights).sum(axis=1)

            # Add Lambda_2 to Lambda
            Lambda += (theta.sum() * self.ensemble_significance[1])

        log_likelihood = log_sum_nnz - Lambda

        return log_likelihood

    def __get_p_value_fusion_arithmetic(self, p_values, coords):
        '''
        Calculate p_value for instances in dimension 'p_value_fusion_index' using
        p_value fusion with arithmetic mean over the all tensor (including zeros).

        Parameters
        ----------
        p_values : list
            List of Poisson p-values.
        coords : numpy array
            Coordinates of the non-zero elements within the sparse tensor.

        Returns
        -------
        p_value_fusions : list
            p_value for each instance in dimension 'p_value_fusion_index'

        '''

        p_value_fusions = list()

        # unique array of target instances to calculate p_value fussion
        unfold_target = np.unique(coords[:, self.p_value_fusion_index], axis=0)

        # unfold size is the tensor size unfolding at index 'p_value_fusion_index'
        # multiply dimension sizes ignoring the unfolding dimension 'p_value_fusion_index'
        unfold_size = np.delete(self.dimensions_size, self.p_value_fusion_index).prod()

        # calculate p-value for each target
        for target in unfold_target:
            # coordinates of the current target
            nnz_indices = np.where((coords[:, self.p_value_fusion_index] == target).all(axis=1))
            # remove the current target from bag for speed improvment on next iteration
            coords = np.delete(coords, nnz_indices, axis=0)

            target_p_value_sum = p_values[nnz_indices].sum()

            # total number of zero elements stored in the tensor at the current unfolding
            total_zero = unfold_size - len(nnz_indices)

            arithmetic_p_value = (target_p_value_sum + total_zero) / unfold_size

            p_value_fusions.append(arithmetic_p_value)

        return p_value_fusions

    def __get_p_value_fusion_harmonic(self, p_values, coords):
        '''
        Calculate p_value for instances in dimension 'p_value_fusion_index' using
        p_value fusion with harmonic mean over the all tensor (including zeros).

        Parameters
        ----------
        p_values : list
            List of Poisson p-values.
        coords : numpy array
            Coordinates of the non-zero elements within the sparse tensor.

        Returns
        -------
        p_value_fusions : list
            p_value for each instance in dimension 'p_value_fusion_index'

        '''
        p_value_fusions = list()

        # unique array of target instances to calculate p_value fussion
        unfold_target = np.unique(coords[:, self.p_value_fusion_index], axis=0)

        # unfold size is the tensor size unfolding at index 'p_value_fusion_index'
        # multiply dimension sizes ignoring the unfolding dimension 'p_value_fusion_index'
        unfold_size = np.delete(self.dimensions_size, self.p_value_fusion_index).prod()

        # calculate p-value for each target
        for target in unfold_target:
            # coordinates of the current target
            nnz_indices = np.where((coords[:, self.p_value_fusion_index] == target).all(axis=1))
            # remove the current target from bag for speed improvment on next iteration
            coords = np.delete(coords, nnz_indices, axis=0)

            target_p_value_sum = (1 / p_values[nnz_indices]).sum()

            # total number of zero elements stored in the tensor at the current unfolding
            total_zero = unfold_size - len(nnz_indices)

            harmonic_p_value = unfold_size / (target_p_value_sum + total_zero)

            p_value_fusions.append(harmonic_p_value)

        return p_value_fusions

    def __get_p_value_fusion_harmonic_observed(self, p_values, coords):
        '''
        Calculate p_value for instances in dimension 'p_value_fusion_index' using
        p_value fusion with harmonic mean over the observed events only.

        Parameters
        ----------
        p_values : list
            List of Poisson p-values.
        coords : numpy array
            Coordinates of the non-zero elements within the sparse tensor.

        Returns
        -------
        p_value_fusions : list
            p_value for each instance in dimension 'p_value_fusion_index'

        '''
        p_value_fusions = list()

        # unique array of target instances to calculate p_value fussion
        unfold_target = np.unique(coords[:, self.p_value_fusion_index], axis=0)

        # calculate p-value for each target
        for target in unfold_target:
            # coordinates of the current target
            nnz_indices = np.where((coords[:, self.p_value_fusion_index] == target).all(axis=1))
            # remove the current target from bag for speed improvment on next iteration
            coords = np.delete(coords, nnz_indices, axis=0)

            target_p_value_sum = (1 / p_values[nnz_indices]).sum()

            harmonic_p_value = len(nnz_indices) / target_p_value_sum

            p_value_fusions.append(harmonic_p_value)

        return p_value_fusions

    def __get_p_value_fusion_chi2_observed(self, p_values, coords):
        '''
        Calculate p-value fusion using Fisher's test over observed events

        Parameters
        ----------
        p_values : list
            List of Poisson p-values.
        coords : numpy array
            Coordinates of the non-zero elements within the sparse tensor.

        Returns
        -------
        p_value_fusions : list
            p_value for each instance in dimension 'p_value_fusion_index'

        '''
        p_value_fusions = list()

        # unique array of target instances to calculate p_value fussion
        unfold_target = np.unique(coords[:, self.p_value_fusion_index], axis=0)

        # calculate p-value for each target
        for target in unfold_target:
            # coordinates of the current target
            nnz_indices = np.where((coords[:, self.p_value_fusion_index] == target).all(axis=1))
            # remove the current target from bag for speed improvment on next iteration
            coords = np.delete(coords, nnz_indices, axis=0)

            statistic = -2 * (np.log(p_values[nnz_indices]).sum())

            k = len(nnz_indices)
            # chi2_p_value = chi2.cdf(2*k, statistic)
            chi2_p_value = chi2_cdf(2 * k, statistic)

            p_value_fusions.append(chi2_p_value)

        return p_value_fusions

    def __get_p_value_fusion_chi2(self, p_values, coords):
        '''
        Calculate p-value fusion using Fisher's test

        Parameters
        ----------
        p_values : list
            List of Poisson p-values.
        coords : numpy array
            Coordinates of the non-zero elements within the sparse tensor.

        Returns
        -------
        p_value_fusions : list
            p_value for each instance in dimension 'p_value_fusion_index'

        '''
        p_value_fusions = list()

        # unique array of target instances to calculate p_value fussion
        unfold_target = np.unique(coords[:, self.p_value_fusion_index], axis=0)

        # unfold size is the tensor size unfolding at index 'p_value_fusion_index'
        # multiply dimension sizes ignoring the unfolding dimension 'p_value_fusion_index'
        unfold_size = np.delete(self.dimensions_size, self.p_value_fusion_index).prod()

        # calculate p-value for each target
        for target in unfold_target:
            # coordinates of the current target
            nnz_indices = np.where((coords[:, self.p_value_fusion_index] == target).all(axis=1))
            # remove the current target from bag for speed improvment on next iteration
            coords = np.delete(coords, nnz_indices, axis=0)

            statistic = -2 * (np.log(p_values[nnz_indices]).sum())

            # chi2_p_value = chi2.cdf(2*k, statistic)
            chi2_p_value = chi2_cdf(2 * unfold_size, statistic)

            p_value_fusions.append(chi2_p_value)

        return p_value_fusions

    def __get_p_values(self, lambdas, values):
        '''
        Calculate the Poisson p-value for the given x (number of occurrences values[i])
        and Poisson lambda (lambdas[i])

        Parameters
        ----------
        lambdas : numpy array
            array of Poisson lambdas.
        values : numpy array
            non-zero values of a tensor.

        Returns
        -------
        p_values : list
            List of Poisson p-values.

        '''

        p_value_calculator = lambda value, lamb: (1 - poisson.cdf(int(value) - 1, lamb))
        p_value_func = np.vectorize(p_value_calculator)

        p_values = p_value_func(values, lambdas)

        return p_values

    def __get_lambdas(self, coords):
        '''
        Calculate lambda, average number of occurences for Poisson Distribution.

        Parameters
        ----------
        coords : numpy array
            Coordinates of the non-zero elements within the sparse tensor.

        Returns
        -------
        lambdas : numpy array
            array of Poisson lambda values.

        '''
        theta = 1
        for dimension_index, dimension in enumerate(self.dimensions):
            if dimension_index not in self.ignore_dimensions_indx:
                theta *= (dimension[coords[:, dimension_index]]) * self.mode_weights[dimension_index]
        theta *= self.weights

        if self.K == 1:  # if rank is 1, no components to sum
            lambdas = theta
        else:
            lambdas = theta.sum(axis=1)

        if self.lambda_method == 'ensemble':
            # main ktensor's lambdas weight
            lambdas *= self.ensemble_significance[0]

            # calculate the secondary tensor's lambdas
            theta = 1
            for dimension_index, dimension in enumerate(self.ensemble_dimensions):
                if dimension_index not in self.ignore_dimensions_indx:
                    theta *= dimension[coords[:, dimension_index]]
            theta *= self.ensemble_weights

            # combine lambdas
            if self.K_ensemble == 1:  # if rank is 1, no components to sum
                lambdas += (theta * self.ensemble_significance[1])
            else:
                lambdas += (theta.sum(axis=1) * self.ensemble_significance[1])

        return lambdas
