#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of permute utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""

def permute(M, order):
    """
    This function permutes the dimensions of the KRUSKAL tensor M.

    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    order : array
        Vector order.
        
    Returns
    -------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    """

    for ii, dim in enumerate(order):
        temp = M.Factors[str(ii)]
        M.Factors[str(ii)] = M.Factors[str(dim)]
        M.Factors[str(dim)] = temp

    return M