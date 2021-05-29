#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tensor_anomaly_detection_v2.py performs p-value scoring over the tensor decomposition, i.e.
the KRUSKAL tensor M. The calculated p-values are used to detect anomalies.\n
This method was introduced by Eren et al. in [1].\n
The second version performs faster calculation of the inner products of the
components to extract the lambdas.\n
This version also provides dimension fusion methods for lambda calculations.

References
========================================
[1] M. E. Eren, J. S. Moore and B. S. Alexandro, "Multi-Dimensional Anomalous Entity Detection via Poisson Tensor Factorization," 2020 IEEE International Conference on Intelligence and Security Informatics (ISI), 2020, pp. 1-6, doi: 10.1109/ISI49825.2020.9280524.

@author: Juston S. Moore, Maksim Ekin Eren
"""
from . ktensor_utils import get_X_hat
from . ktensor_utils import get_X_size
import numpy as np
import numpy_indexed as npi
from collections import OrderedDict
from scipy.special import logsumexp
from scipy.stats import combine_pvalues
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pandas as pd


class PoissonTensorAnomaly_v2():

    def __init__(self, components, indicies, tensor_weights=[1]):
        """
        Initilize the anomaly detection class.\n
        Calculates the lambdas, and obtains tensor information.

        Parameters
        ----------
        components : dict
            KRUSKAL Tensor M in dict format.
        indicies : array
            Non-zero coordinates.
        tensor_weights : list, optional
            Weight of each lambda for the tensors.\n
            Used only when ensemble of tensors used in lambda calculations.
            The default is [1].

        """

        temp_lamb = list()
        for ii, tw in enumerate(tensor_weights):
            l = get_X_hat(components[ii], indicies)
            temp_lamb.append(l * tensor_weights[ii])

        self.lambdas = sum(temp_lamb)
        self.indices = indicies
        self.tensor_shape = get_X_size(components)

        self.fusion_scores = None
        self.link_prediction_scores = dict()
        self.fusion_preds = dict()

    def get_lambdas(self):
        """
        Returns the lambda values that are calculated.

        Returns
        -------
        lambdas : array
            Array of lambda values for the indices.

        """
        return self.lambdas

    def get_link_prediction_scores(self, y):
        """
        Calculates the prediction scores given lambdas and the true labels y.

        Parameters
        ----------
        y : list
            True labels.\n
            Label of each index.

        Returns
        -------
        score : dict
            Prediction scores. {"roc_auc": float, "pr_auc": float}

        """

        roc_auc = roc_auc_score(y, -self.lambdas)
        precision, recall, _ = precision_recall_curve(y, -self.lambdas, pos_label=1)
        pr_auc =  auc(recall, precision)
        self.link_prediction_scores['roc_auc'] = roc_auc
        self.link_prediction_scores['pr_auc'] = pr_auc

        return self.link_prediction_scores

    def get_dimension_fusion_scores(self, axis_map, y_true):
        """
        Calculates the prediction scores given fuzed lambdas and the true labels y.\n
        Fusion is performed for the dimension in axis_map.

        Parameters
        ----------
        axis_map : list
            Which dimensions to fuse.
        y_true : list
            List of true labels for each entry.

        Returns
        -------
        df : Pandas DataFrame
            Fusion scores.

        """
        log_pval = logsumexp(a=[-self.lambdas, np.zeros_like(self.lambdas)], b=np.array([[-1., 1.]] * self.lambdas.shape[0]).T, axis=0)

        self.fusion_preds['harmonic_allLinks'] = dict()
        self.fusion_preds['harmonic_observedLinks'] = dict()
        self.fusion_preds['fisher_allLinks'] = dict()
        self.fusion_preds['fisher_observedLinks'] = dict()

        results = OrderedDict()
        results = OrderedDict([
        (k, {
            'y_true': dict(zip(
                ('keys', 'labels'),
                zip(*npi.group_by(keys=self.indices[:, axes], values=y_true, reduction=np.max))
            )),
            'y_pred': OrderedDict([
                ('harmonic_allLinks',      self.__fuse_pval(self.indices, log_pval, tensor_shape=self.tensor_shape, fuse_to_axes=axes, fuser=self.__log_harmonic_mean, set_N=True)),
                #('harmonic_observedLinks', self.__fuse_pval(self.indices, log_pval, tensor_shape=self.tensor_shape, fuse_to_axes=axes, fuser=self.__log_harmonic_mean, set_N=False)),
                #('fisher_allLinks',        self.__fuse_pval(self.indices, log_pval, tensor_shape=self.tensor_shape, fuse_to_axes=axes, fuser=self.__fisher_fusion, set_N=True)),
                #('fisher_observedLinks',   self.__fuse_pval(self.indices, log_pval, tensor_shape=self.tensor_shape, fuse_to_axes=axes, fuser=self.__fisher_fusion, set_N=False))
            ])
        })
        for k, axes in axis_map.items()
        ])

        y_true = OrderedDict()

        for fusion in axis_map.keys():
            keys_0 = np.array(results[fusion]['y_true']['keys'])
            assert np.all(np.array(results[fusion]['y_true']['keys']) == keys_0)
            labels_0 = np.array(results[fusion]['y_true']['labels'])
            assert np.all(np.array(results[fusion]['y_true']['labels']) == labels_0)

            y_true[fusion] = labels_0

        performance = {}

        for fusion, v2 in results.items():

            for method, y_pred in v2['y_pred'].items():
                # Remove any scores of -inf
                mask = np.isneginf(y_pred)
                y_pred[mask] = np.min(y_pred[~mask]).min() - 10.

                performance[(fusion, method, 'ROC-AUC')] = roc_auc_score(y_true[fusion], -y_pred)
                precision, recall, _ = precision_recall_curve(y_true[fusion], -y_pred, pos_label=1)
                performance[(fusion, method, 'PR-AUC')] = auc(recall, precision)
                self.fusion_preds[str(method)][str(fusion)] = dict()
                self.fusion_preds[str(method)][str(fusion)]['y_pred'] = y_pred
                self.fusion_preds[str(method)][str(fusion)]['y_true'] = y_true[fusion]


        df = pd.DataFrame(
            data=[
                list(k) + [v]
                for k, v in performance.items()
                    ],
                    columns=[
                        'fusion',
                        'method',
                        'metric',
                        'score'
                    ]
            )

        self.fusion_scores = df
        return df

    def __preproc_dec(fn):
        """
        Wraps the log array into a function to be used in fusion.

        Parameters
        ----------
        fn : function
            Function to be wrapped.

        Returns
        -------
        function
            Wrapped function to be used during fusion.

        """
        def wrap(self, log_arr, axis=None, N=None, **kwargs):

            log_arr = np.asarray(log_arr, dtype=np.float64)
            if axis is None:
                assert log_arr.ndim == 1
                axis = 0

            if N is None:
                N = log_arr.shape[axis]

            N_extra = N - log_arr.shape[axis]
            assert N_extra >= 0

            return fn(self, log_arr, axis=axis, N=N, N_extra=N_extra, **kwargs)
        return wrap

    @__preproc_dec
    def __log_harmonic_mean(self, log_arr, axis, N, N_extra):
        """
        Fuses dimensions using Log of Harmonic Mean.

        Parameters
        ----------
        log_arr : array
            Log of lambdas.
        axis : list
            which dimensions to fuse.
        N : int
            Number of elements in current dimension.
        N_extra : int
            Number of elements in other dimensions.

        Returns
        -------
        array
            Harmonic mean fusion results.

        """
        if N_extra > 0:
            sh = list(log_arr.shape)
            sh[axis] = 1
            log_arr = np.concatenate((log_arr, np.ones(sh) * np.log(N_extra)), axis=axis)

        return np.log(N) - logsumexp(-log_arr, axis=axis)

    @__preproc_dec
    def __fisher_fusion(self, log_arr, axis, N, N_extra):
        """
        Fuses dimensions using Fisher Fusion.

        Parameters
        ----------
        log_arr : array
            Log of lambdas.
        axis : list
            which dimensions to fuse.
        N : int
            Number of elements in current dimension.
        N_extra : int
            Number of elements in other dimensions.

        Returns
        -------
        array
            Fisher fusion results.

        """
        statistic = -2 * np.sum(log_arr, axis=axis)
        return chi2.logsf(statistic, 2 * N)

    def __fuse_pval(self, indices, log_pval, fuse_to_axes, fuser, set_N, tensor_shape):
        """
        Perform p-value fusion given the indices and the log of p-values.

        Parameters
        ----------
        indices : array
            Array of non-zero indices.
        log_pval : array
            Log of p-values for the given indices.
        fuse_to_axes : list
            which modes to fuse.
        fuser : function
            Fusion function that is used.
        set_N : bool
            If true, calculates fusion for all links. Otherwise, calculates only for the observed links.
        tensor_shape : array
            Shape of the tensor X, i.e. size of each mode.

        Returns
        -------
        array
            p-value fusion of the dimensions.

        """
        if set_N:
            raw_fuser = fuser
            tensor_shape = np.array(tensor_shape)
            N = np.prod(tensor_shape) / np.prod(tensor_shape[fuse_to_axes])
            fuser = lambda log_arr: raw_fuser(log_arr, N=N)

        result = npi.group_by(keys=indices[:, fuse_to_axes], values=log_pval, reduction=fuser)

        return np.array([x[1] for x in result])
