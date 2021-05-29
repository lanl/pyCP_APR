#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stat_utils.py contains the tensor statistic utilities.

@author: Maksim Ekin Eren
"""
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


def mrr_fuse_ranks(x, weights=None, axis=0, k=60., y=None):
    """
    Calculates Mean Reciprocal Rank (MRR).\n
    Under development.

    Parameters
    ----------
    x : array
        Tensor x.
    weights : array, optional
        Array of weights. The default is None.
    axis : int, optional
        Dimension number. The default is 0.
    k : int, optional
        Top k. The default is 60..
    y : array, optional
        Labels. The default is None.

    Returns
    -------
    result : float
        MRR score.

    """

    x = np.asarray(x)
    rank = x.argsort(axis=axis).argsort(axis=axis)

    y_pred = (1. / (k + rank)).mean(axis=0)

    result = dict()
    result['y_pred'] = y_pred

    if y is not None:
        result['ROC-AUC'] = roc_auc_score(y, y_pred)
        precision, recall, _ = precision_recall_curve(y, y_pred, pos_label=1)
        result['PR-AUC'] = auc(recall, precision)

    return result
