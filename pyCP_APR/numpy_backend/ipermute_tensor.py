#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of ipermute utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import numpy as np

def ipermute(X, order):
    """
    This function inverse-permutes the dimensions of X.

    Parameters
    ----------
    X : dense tensor
        Dense tensor object X.
    order : array
        Vector order.
        
    Returns
    -------
    X : dense tensor
        Inverse permute of X by the order specified.

    """
    X.data = np.transpose(X.data, np.argsort(order))
    X.Size = list(X.data.shape)

    return X