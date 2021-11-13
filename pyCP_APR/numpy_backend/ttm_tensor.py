#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of ttm utility with Numpy backend from the MATLAB Tensor Toolbox [1].

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
"""
import numpy as np
import copy

from . tt_dimscheck import tt_dimscheck
from . tensor import TENSOR
from . ipermute_tensor import ipermute

def ttm(X, V, varargin={}):
    """
    Tensor times matrix operation.

    Parameters
    ----------
    X : object
        Sparse tensor. sptensor.SP_TENSOR.
    V : np.ndarray
        Numpy array.
    varargin : dict
        Optional parameter to specify tflag and or mode settings.
        
    Returns
    -------
    Y : object
        Sparse tensor. sptensor.SP_TENSOR.
    """
    #
    # Create 'n' and 'tflag' arguments from varargin
    #
    n = np.arange(0, X.Dimensions)
    tflag = ''
    ver = 0
    
    if len(varargin) == 1:
        if 'tflag' in varargin:
            tflag = varargin['tflag']
        else:
            n = varargin['n']
        
    elif len(varargin) == 2:
        n = varargin['n']
        tflag = varargin['tflag']

    elif len(varargin) == 3:
        n = varargin['n']
        tflag = varargin['tflag']
        ver = varargin['ver']
    
    #
    # Handle cell array
    #
    if isinstance(V, list):
        dims = n
        dims, vidx = tt_dimscheck(dims, X.Dimensions, len(V))
        Y = ttm(copy.deepcopy(X), V[vidx[0]], varargin={'n':dims[0], 'tflag':tflag})
        for k in range(1, len(dims)):
            Y = ttm(copy.deepcopy(Y), V[vidx[k]], varargin={'n':dims[k], 'tflag':tflag})
    
    else:
        #
        # COMPUTE SINGLE N-MODE PRODUCT 
        #
        N = X.Dimensions
        sz = X.Size
        
        if ver == 0:
            order = [n] + list(np.arange(0, n)) + list(np.arange(n+1, N))
            newdata = np.transpose(X.data.copy(), order)
            newdata = np.reshape(newdata, (sz[n], np.prod(sz[0:n] + sz[n+1:N])))
            
            if tflag == 't':
                newdata = np.dot(V.T, newdata)
                p = V.shape[1]
            else:
                newdata = np.dot(V, newdata)
                p = V.shape[0]
                
            newsz = [p] + sz[0:n] + sz[n+1:N]
            newdata = np.reshape(newdata, newsz)
            Y = TENSOR(newdata)
            Y = ipermute(Y, order)
            
        else:
            raise Exception("Not yet implemented!") 
    
    return Y
        
    
