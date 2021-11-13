#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sptensor.py contains the SP_TENSOR class which is the object representation
of the sparse tensor X in COO format.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
import sys
import numpy as np
import sparse

class SP_TENSOR():

    def __init__(self, Coords, Values, Size=[]):
        """
        Initilize the SP_TENSOR class.\n
        Sorts the tensor entries.

        Parameters
        ----------
        Coords : Numpy array (i.e. array that is a list of list)
            Array of non-zero coordinates for sparse tensor X. COO format.\n
            Each entry in this array is a coordinate of a non-zero value in the tensor.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            len(Coords) is number of total entiries in X, and len(coords[0]) should give the number of dimensions.
        Values : Numpy array (i.e. list of non-zero values corresponding to each list of non-zero coordinates)
            Array of non-zero tensor entries. COO format.\n
            Used when Type = 'sptensor' and tensor parameter is not passed.\n
            Length of values must match the length of coords.
        Size : list
            Optional parameter to specify the size of the sparse tensor.

        """
        self.Dimensions = Coords.shape[1]
        self.Coords, self.data = self.__sort_coords(Coords, Values)
        self.Type = 'sptensor'
        self.Size = []
        
        if len(Size) == 0:
            for n in range(self.Dimensions):
                self.Size.append(max(self.Coords[:,n])+1)
        else:
            self.Size = Size
        
    
    def todense(self):
        X = sparse.COO(self.Coords.T, self.data, shape=tuple(self.Size))
        return X.todense()

    def __sort_coords(self, Coords, Values):
        """
        Helper function to sort the COO representation of the tensor.

        Parameters
        ----------
        Coords : array
            Coordinates of non-zero values.
        Values : array
            List of values for each coordinate.

        Returns
        -------
        Coords : array
            Sorted coordinates of non-zero values..
        Values : array
            Sorted list of values for each coordinate..

        """

        for d in range(self.Dimensions, 0, -1):
            if d == self.Dimensions:
                sort_indices = Coords[:, d - 1].argsort()

            else:
                sort_indices = Coords[:, d - 1].argsort(kind='mergesort')

            Coords = Coords[sort_indices]
            Values = Values[sort_indices]

        return Coords, Values
