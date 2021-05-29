#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sptensor_Torch.py contains the SP_TENSOR class which is the object representation
of the sparse tensor X in COO format.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: maksimekineren
"""
import sys
import numpy as np
import torch as tr


class SP_TENSOR():

    def __init__(self, Tensor, Coords, Values, dtype='torch.DoubleTensor', device='cpu'):
        """
        Initilize the SP_TENSOR class.\n
        Sorts the tensor entries.

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
        dtype : string, optional
            Type to be used in torch tensors.
            Default is torch.cuda.DoubleTensor.
        device :string, optional
            Torch device to be used.
            'cpu' to use PyTorch with CPU.
            'gpu' to use cuda:0
            The default is cpu.

        """

        self.Type = 'sptensor'
        self.dtype = dtype
        self.device = device

        # Sparse PyTorch Tensor is passed
        if tr.is_tensor(Tensor):
            self.Size = Tensor.size()
            self.Dimensions = Tensor.dim()
            self.nnz = Tensor._nnz()
            self.Coords, self.Values = self.__sort_coords(Tensor._indices().numpy(),
                                                          Tensor._values().numpy())

        # Starting with numpy
        else:
            self.Size = list()
            self.Dimensions = Coords.shape[1]
            self.Coords, self.Values = self.__sort_coords(Coords, Values)
            self.nnz = len(Coords)

            for d in range(self.Dimensions):
                self.Size.append((tr.max(self.Coords[:, d]) + 1).data.tolist())

    def ttv(self, vecs):
        """
        Tensor times vector for KRUSKAL tensor M.

        Parameters
        ----------
        vecs : array
            column vector.

        Returns
        -------
        c : array
             product of KRUSKAL tensor X with the vector vecs.

        """

        dims = tr.arange(self.Dimensions)
        vidx = tr.arange(self.Dimensions)

        combined = tr.cat((dims, vidx))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        intersection = uniques[counts > 1]

        remdims = difference

        for d in range(self.Dimensions):
            if len(vecs[str(vidx.data.tolist()[d])]) != self.Size[d]:
                sys.exit('Multiplicand is wrong size')

        newvals = self.Values
        subs = self.Coords

        if len(subs) == 0:
            newsubs = []

        for d in range(self.Dimensions):
            idx = self.Coords[:, d]
            w = vecs[str(vidx.data.tolist()[d])]
            bigw = w[idx]
            newvals = tr.mul(newvals, bigw)

        newsubs = subs[:, remdims]

        if len(remdims) == 0:
            c = tr.sum(newvals)
            return c

        sys.exit("Reached to a location that has not been imlemented yet.")
        return -1

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

        # Convert to pyTorch Tensor
        if isinstance(Coords, (list, np.ndarray)):
            Coords = tr.from_numpy(Coords).to(self.device)

        if isinstance(Values, (list, np.ndarray)):
            Values = tr.from_numpy(Values).type(self.dtype).to(self.device)

        return Coords, Values
