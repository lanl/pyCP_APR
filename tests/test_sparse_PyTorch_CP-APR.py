"""
    Tests CP-APR PyTorch implementation using Sparse Tensor
    
    Run with: python -m unittest test_sparse_PyTorch_CP-APR.py
"""
from pyCP_APR.pyCP_APR import CP_APR
import unittest
import scipy.io as spio
import numpy as np
import torch

class TestPyTorchCP_APR(unittest.TestCase):
    
    def setUp(self):
        """Setup the test."""
        
        # Sparse tensor coordinates and non-zero values
        coords = spio.loadmat('../data/test_data/subs.mat', squeeze_me=True)['values'] - 1
        nnz_binary = spio.loadmat('../data/test_data/vals_binary.mat', squeeze_me=True)['values']
        nnz_count = spio.loadmat('../data/test_data/vals_count.mat', squeeze_me=True)['values']
        
        # Initial factor values
        M_init = {"Factors":{}}
        dim = 0
        for key, values in spio.loadmat('../data/test_data/minit.mat', squeeze_me=True).items():
            if 'init_f' in key:
                M_init["Factors"][str(dim)] = values
                dim += 1
                
        # Expected latent factors and lambda values
        M_expected_binary = dict()
        dimension = 0
        for key, values in spio.loadmat('../data/test_data/m_expected_binary.mat', squeeze_me=True).items():
            if 'm_' in key:
                M_expected_binary[str(dimension)] = values
                dimension += 1
            if 'lambd' in key:
                M_expected_binary['lambda'] = values
                
        M_expected_count = dict()
        dimension = 0
        for key, values in spio.loadmat('../data/test_data/m_expected_count.mat', squeeze_me=True).items():
            if 'm_' in key:
                M_expected_count[str(dimension)] = values
                dimension += 1
            if 'lambd' in key:
                M_expected_count['lambda'] = values
                
        self.sparse = dict()
        self.sparse['coords'] = coords
        self.sparse['nnz_binary'] = nnz_binary
        self.sparse['nnz_count'] = nnz_count
        self.sparse['M_init'] = M_init
        self.sparse['M_expected_binary'] = M_expected_binary
        self.sparse['M_expected_count'] = M_expected_count
        
        # Initilize CP-APR
        self.cp_apr_cpu = CP_APR(n_iters=1000, verbose=0, method='torch', device='cpu')
        
        # Make sure CUDA device is available
        if torch.cuda.is_available():
            self.cp_apr_gpu = CP_APR(n_iters=1000, verbose=0, device='gpu', method='torch')
        
    def take_norm_diff_factor(self, decomposition, d, M_type):
        """Helper function to take norm difference between two factors."""
        
        pred_di = decomposition['Factors'][str(d)]
        expected_di = self.sparse[M_type][str(d)]
        norm_diff_di = np.linalg.norm(pred_di - expected_di)

        return norm_diff_di
    
    
    def take_norm_diff_weights(self, decomposition, M_type):
        """Helper function to take norm difference between two weight of factors."""
        
        pred_lambd =  decomposition['Weights']
        expected_lambd = self.sparse[M_type]['lambda']
        norm_diff_lambd = np.linalg.norm(np.array(pred_lambd) - np.array(expected_lambd))
        
        return norm_diff_lambd
        
        
    def test_latent_factors_binary(self):
        """Make sure the resulting latent factors are as expected for binary tensor."""
                
        decomposition = self.cp_apr_cpu.fit(coords=self.sparse['coords'], 
                                          values=self.sparse['nnz_binary'], 
                                          rank=2, Minit=self.sparse['M_init'])
        
        # Check each latent factor
        for d in range(len(self.sparse['coords'][0])):
            
            norm_diff_di = self.take_norm_diff_factor(decomposition, d, 'M_expected_binary')
            
            # check if norm of difference is very small
            self.assertEqual(True ,(np.abs(norm_diff_di) < 0.001))
            
       # Compare the weights
        norm_diff_lambd = self.take_norm_diff_weights(decomposition, 'M_expected_binary')
        
        # check if norm of difference is very small
        self.assertEqual(True ,(np.abs(norm_diff_lambd) < 0.001))

            
    def test_latent_factors_count(self):
        """Make sure the resulting latent factors are as expected for count tensor."""
        
        decomposition = self.cp_apr_cpu.fit(coords=self.sparse['coords'], 
                                          values=self.sparse['nnz_count'], 
                                          rank=2, Minit=self.sparse['M_init'])
        
        # Check each latent factor
        for d in range(len(self.sparse['coords'][0])):
            
            norm_diff_di = self.take_norm_diff_factor(decomposition, d, 'M_expected_count')
            
            # check if norm of difference is very small
            self.assertEqual(True ,(np.abs(norm_diff_di) < 0.001))
            
       # Compare the weights
        norm_diff_lambd = self.take_norm_diff_weights(decomposition, 'M_expected_count')
        
        # check if norm of difference is very small
        self.assertEqual(True ,(np.abs(norm_diff_lambd) < 0.001))
        
        
    def test_latent_factors_binary_GPU(self):
        """Make sure the resulting latent factors are as expected for binary tensor. Tests using GPU."""
        
        # Runs if CUDA available.
        if torch.cuda.is_available():
            decomposition = self.cp_apr_gpu.fit(coords=self.sparse['coords'], 
                                              values=self.sparse['nnz_binary'], 
                                              rank=2, Minit=self.sparse['M_init'])

            # Check each latent factor
            for d in range(len(self.sparse['coords'][0])):

                norm_diff_di = self.take_norm_diff_factor(decomposition, d, 'M_expected_binary')

                # check if norm of difference is very small
                self.assertEqual(True ,(np.abs(norm_diff_di) < 0.001))

           # Compare the weights
            norm_diff_lambd = self.take_norm_diff_weights(decomposition, 'M_expected_binary')

            # check if norm of difference is very small
            self.assertEqual(True ,(np.abs(norm_diff_lambd) < 0.001))

            
    def test_latent_factors_count_GPU(self):
        """Make sure the resulting latent factors are as expected for count tensor. Tests using GPU."""

        # Runs if CUDA available.
        if torch.cuda.is_available():
            decomposition = self.cp_apr_gpu.fit(coords=self.sparse['coords'], 
                                              values=self.sparse['nnz_count'], 
                                              rank=2, Minit=self.sparse['M_init'])

            # Check each latent factor
            for d in range(len(self.sparse['coords'][0])):

                norm_diff_di = self.take_norm_diff_factor(decomposition, d, 'M_expected_count')

                # check if norm of difference is very small
                self.assertEqual(True ,(np.abs(norm_diff_di) < 0.001))

           # Compare the weights
            norm_diff_lambd = self.take_norm_diff_weights(decomposition, 'M_expected_count')

            # check if norm of difference is very small
            self.assertEqual(True ,(np.abs(norm_diff_lambd) < 0.001))
