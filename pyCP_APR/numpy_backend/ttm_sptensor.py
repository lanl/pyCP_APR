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

from . sptensor import SP_TENSOR
from . tt_dimscheck import tt_dimscheck
from . tensor import TENSOR
from . tenmat_sptensor import tenmat

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
        if tflag == 't':
            V = V.T
            
        cdims = [n]
        rdims = list(set(list(np.arange(0, X.Dimensions))) - set([n]))
        
        siz = list(X.Size)
        siz[n] = V.shape[0]
        siz = np.array(siz)
        
        order = rdims + cdims
        
        Xnt = tenmat(X, n)
        Z = np.dot(Xnt.T, V.T)
        
        c = np.transpose(np.nonzero(Z))
        nnz_values = Z[np.nonzero(Z)]
        
        a = np.transpose(np.unravel_index(c[:,0], siz[rdims]))
        b = np.transpose(np.unravel_index(c[:,1], siz[cdims]))
        
        new_order = [0] * len(order)
        for old_idx, new_idx in enumerate(order):
            new_order[new_idx] = old_idx
        
        nnz_coords = np.hstack([a,b])[:,new_order]
        
        Y = SP_TENSOR(nnz_coords, nnz_values, siz)
        
    
    return Y
        
