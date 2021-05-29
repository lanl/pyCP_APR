import scipy.io as spio
import numpy as np
from pyCP_APR import CP_APR

#
# Load data
#
X = spio.loadmat('../data/test_data/subs.mat', squeeze_me=True)['values'] - 1
vals = spio.loadmat('../data/test_data/vals_count.mat', squeeze_me=True)['values']


#
# Extract the initial KRUSKAL tensor M
#
M_init = dict()
dim = 0
for key, values in spio.loadmat('../data/test_data/minit.mat', squeeze_me=True).items():
    if 'init_f' in key:
        M_init[str(dim)] = values
        dim += 1
        
#
# Run CP-APR
#
cp_apr = CP_APR(n_iters=100, verbose=10, method='torch', device='cpu')
result = cp_apr.fit(coords=X, values=vals, rank=2, Minit=M_init)


#
# Look at the results
# 
print(result)