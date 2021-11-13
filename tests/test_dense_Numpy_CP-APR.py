"""
    Tests CP-APR Numpy implementation using Dense Tensor
    
    Run with: python -m unittest test_dense_Numpy_CP-APR.py
"""
from pyCP_APR.pyCP_APR import CP_APR
import unittest
import scipy.io as spio
import numpy as np


class TestNumpyCP_APR(unittest.TestCase):

    def setUp(self):
        """Setup the test."""
        
        # Sparse tensor coordinates and non-zero values
        
        # Initial factor values
        M_init = {"Factors":{}, "Weights":[1,1]}
        dim = 0
        for key, values in spio.loadmat('../data/test_data/minit.mat', squeeze_me=True).items():
            if 'init_f' in key:
                M_init["Factors"][str(dim)] = values
                dim += 1
                
        # Expected latent factors and lambda values
        M_expected_binary = dict()
        dimension = 0
        for key, values in spio.loadmat('../data/test_data/m_expected_binary_dense.mat', squeeze_me=True).items():
            if 'm_' in key:
                M_expected_binary[str(dimension)] = values
                dimension += 1
            if 'lambd' in key:
                M_expected_binary['lambda'] = values
                
                
        M_expected_count = dict()
        dimension = 0
        for key, values in spio.loadmat('../data/test_data/m_expected_count_dense.mat', squeeze_me=True).items():   
            if 'm_' in key:
                M_expected_count[str(dimension)] = values
                dimension += 1
            if 'lambd' in key:
                M_expected_count['lambda'] = values
                
                
        self.dense = dict()
        self.dense['X_binary'] = np.ones((10,30,40))
        self.dense['X_count'] = np.arange(1, 12001).reshape([10,30,40])
        self.dense['M_init'] = M_init
        self.dense['M_expected_binary'] = M_expected_binary
        self.dense['M_expected_count'] = M_expected_count
        
        
        # Initilize CP-APR
        self.cp_apr = CP_APR(n_iters=1000, verbose=0, method='numpy')
        
        
    def take_norm_diff_factor(self, decomposition, d, M_type):
        """Helper function to take norm difference between two factors."""

        pred_di = decomposition['Factors'][str(d)]
        expected_di = self.dense[M_type][str(d)]
        norm_diff_di = np.linalg.norm(pred_di - expected_di)

        return norm_diff_di
    
    
    def take_norm_diff_weights(self, decomposition, M_type):
        """Helper function to take norm difference between two weight of factors."""
        
        pred_lambd =  decomposition['Weights']
        expected_lambd = self.dense[M_type]['lambda']
        norm_diff_lambd = np.linalg.norm(np.array(pred_lambd) - np.array(expected_lambd))
        
        return norm_diff_lambd
        
        
    def test_latent_factors_binary(self):
        """Make sure the resulting latent factors are as expected for binary tensor."""
                
        decomposition = self.cp_apr.fit(tensor=self.dense['X_binary'], 
                                          rank=2, 
                                          Minit=self.dense['M_init'],
                                          Type='tensor')
           # Check each latent factor
        for d in range(3):
            
            norm_diff_di = self.take_norm_diff_factor(decomposition, d, 'M_expected_binary')
            
            # check if norm of difference is very small
            self.assertEqual(True ,(np.abs(norm_diff_di) < 10**-3))
            
        # Compare the weights
        norm_diff_lambd = self.take_norm_diff_weights(decomposition, 'M_expected_binary')
        
        # check if norm of difference is very small
        self.assertEqual(True ,(np.abs(norm_diff_lambd) < 1))
        
        
    def test_latent_factors_count(self):
        """Make sure the resulting latent factors are as expected for count tensor."""
                
        decomposition = self.cp_apr.fit(tensor=self.dense['X_count'], 
                                          rank=2, 
                                          Minit=self.dense['M_init'],
                                          Type='tensor')
        
        # Check each latent factor
        for d in range(3):
            
            norm_diff_di = self.take_norm_diff_factor(decomposition, d, 'M_expected_count')
            # check if norm of difference is very small
            self.assertEqual(True ,(np.abs(norm_diff_di) < 10**-3))
            
        # Compare the weights
        norm_diff_lambd = self.take_norm_diff_weights(decomposition, 'M_expected_count')
        # check if norm of difference is very small
        self.assertEqual(True ,(np.abs(norm_diff_lambd) < 1))
        
     