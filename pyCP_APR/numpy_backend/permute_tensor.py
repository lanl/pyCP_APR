#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of permute utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import numpy as np

def permute(X, order):
    """
    This function permutes the dimensions of X.

    Parameters
    ----------
    X : object
        Dense tensor class. tensor.TENSOR.
    order : array
        Vector order.
        
    Returns
    -------
    X : object
        Dense tensor class. tensor.TENSOR.
    """
    X.data = np.transpose(X.data, order)
    X.Size = list(X.data.shape)

    return X