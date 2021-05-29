# pyCP_APR

<div align="center", style="font-size: 50px">

[![Build Status](https://github.com/lanl/pyCP_APR/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/lanl/pyCP_APR/actions/workflows/ci_tests.yml/badge.svg?branch=main) [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg) [![Python Version](https://img.shields.io/badge/python-v3.8.5-blue)](https://img.shields.io/badge/python-v3.8.5-blue) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.4840598-blue.svg)](https://doi.org/10.5281/zenodo.4840598)

</div>

**pyCP_APR** is a Python library for tensor decomposition and anomaly detection. It is designed for the fast analysis of large datasets by accelerating computation speed using GPUs. pyCP_APR uses the CANDECOMP/PARAFAC Alternating Poisson Regression (CP-APR) tensor factorization algorithm utilizing both Numpy and PyTorch backend. While the Numpy backend can be used for the analysis of both sparse and dense tensors, PyTorch backend provides faster decomposition of large and sparse tensors on the GPU. pyCP_APR's Scikit-learn like API allows comfortable interaction with the library, and include the methods for anomaly detection via the p-values obtained from the CP-APR factorization. The anomaly detection methods via the p-values optained from CP-APR was introduced by Eren et al. in [6] using the [Unified Host and Network Dataset](https://csr.lanl.gov/data/2017/) [5]. Our work follows the [MATLAB Tensor Toolbox](https://www.tensortoolbox.org/cp.html) [1-3] implementation of CP-APR [4].


<div align="center", style="font-size: 50px">

### [:information_source: Documentation](https://lanl.github.io/pyCP_APR/) &emsp; [:orange_book: Example Notebooks](examples/) &emsp; [:bar_chart: Datasets](data/tensors) &emsp; [:page_facing_up: Paper](https://ieeexplore.ieee.org/abstract/document/9280524)

</div>


## Installation

#### Option 1: Install using *pip*
```shell
pip install git+https://github.com/lanl/pyCP_APR.git
```

#### Option 2: Install from source
```shell
git clone https://github.com/lanl/pyCP_APR.git
cd pyCP_APR
conda create --name pyCP_APR python=3.8.5
source activate pyCP_APR
python setup.py install
```

## Prerequisites:
- [Anaconda](https://docs.anaconda.com/anaconda/install/)(Optional)
- numpy>=1.19.2
- numpy-indexed>=0.3.5
- pandas>=1.0.5
- matplotlib>=3.3.4
- joblib>=1.0.1
- scikit-learn>=0.22.2
- scipy>=1.5.3
- seaborn>=0.11.1
- torch>=1.6.0
- requests>=2.25.1

## Example Usage
```python
from pyCP_APR import CP_APR
from pyCP_APR.datasets import load_dataset

# Load a sample tensor
data = load_dataset(name="TOY")
coords_train, nnz_train = data['train_coords'], data['train_count']
coords_test, nnz_test = data['test_coords'], data['test_binary']

# CP-APR Object with PyTorch backend on a GPU. Transfer the latent factors back to Numpy arrays.
cp_apr = CP_APR(n_iters=10,
                random_state=42,
                verbose=1,
                method='torch',
                device='gpu',
                return_type='numpy')

# Train a rank 45 tensor
M = cp_apr.fit(coords=coords_train, values=nnz_train, rank=45)

# Predict the scores over the trained tensor
y_score = cp_apr.predict_scores(coords=coords_test, values=nnz_test)
```
**See the [examples](examples/) for more.**


## How to Cite pyCP_APR?

```latex

@MISC{Eren2021_pyCP_APR,
  author = {M. E. {Eren} and J. S. {Moore} and E. {Skau} and M. {Bhattarai} and G. {Chennupati} and B. S. {Alexandrov}},
  title = {pyCP_APR},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4840598},
  howpublished = {\url{https://github.com/lanl/pyCP_APR}}
}

@INPROCEEDINGS{Eren2020_ISI,
  author={M. E. {Eren} and J. S. {Moore} and B. S. {Alexandrov}},
  booktitle={2020 IEEE International Conference on Intelligence and Security Informatics (ISI)},
  title={Multi-Dimensional Anomalous Entity Detection via Poisson Tensor Factorization},
  year={2020},
  pages={1-6},
  doi={10.1109/ISI49825.2020.9280524}
}
```

## Authors
- [Maksim Ekin Eren](mailto:maksim@lanl.gov): Advanced Research in Cyber Systems, Los Alamos National Laboratory
- [Juston S. Moore](mailto:jmoore01@lanl.gov): Advanced Research in Cyber Systems, Los Alamos National Laboratory
- [Erik Skau](mailto:ewskau@lanl.gov): Computer, Computational, and Statistical Sciences Division, Los Alamos National Laboratory
- [Manish Bhattarai](mailto:ceodspspectrum@lanl.gov): Theoretical Division, Los Alamos National Laboratory
- Gopinath Chennupati: Computer, Computational, and Statistical Sciences Division, Los Alamos National Laboratory
- [Boian S. Alexandrov](mailto:boian@lanl.gov): Theoretical Division, Los Alamos National Laboratory


## Copyright Notice:
>© 2021. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

**LANL C Number: C21028**

## License:
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Developer Test Suite
Developer test suites are located under [```tests/```](tests/) directory. Tests can be ran from this folder using ```python -m unittest *```.


## References
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.

[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.

[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.

[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

[5] M. Turcotte, A. Kent and C. Hash, “Unified Host and Network Data Set”, in Data Science for Cyber-Security. November 2018, 1-22.

[6] M. E. Eren, J. S. Moore and B. S. Alexandrov, "Multi-Dimensional Anomalous Entity Detection via Poisson Tensor Factorization," 2020 IEEE International Conference on Intelligence and Security Informatics (ISI), 2020, pp. 1-6, doi: 10.1109/ISI49825.2020.9280524.
