.. pyCP_APR documentation master file, created by
   sphinx-quickstart on Sat Apr  3 03:51:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyCP_APR's documentation!
========================================

.. image:: ../rd100.png
   :width: 324px
   :height: 200px
   :scale: 100 %
   :alt: RD100
   :align: center

**pyCP_APR** is a Python library for tensor decomposition and anomaly detection that is developed as part of the R&D 100 award wining `SmartTensors AI <https://smart-tensors.lanl.gov/software/>`_ project. It is designed for the fast analysis of large datasets by accelerating computation speed using GPUs. pyCP_APR uses the CANDECOMP/PARAFAC Alternating Poisson Regression (CP-APR) tensor factorization algorithm utilizing both Numpy and PyTorch backend. While the Numpy backend can be used for the analysis of both sparse and dense tensors, PyTorch backend provides faster decomposition of large and sparse tensors on the GPU. pyCP_APR's Scikit-learn like API allows comfortable interaction with the library, and include the methods for anomaly detection via the p-values obtained from the CP-APR factorization. The anomaly detection methods via the p-values optained from CP-APR was introduced by Eren et al. in :cite:p:`Eren2020_ISI` using the `Unified Host and Network Dataset <https://csr.lanl.gov/data/2017/>`_  :cite:p:`UnifiedHostandNetwork2018`. Our work follows the `MATLAB Tensor Toolbox <https://www.tensortoolbox.org/cp.html>`_  :cite:p:`TTB_Software,Bader2006,Bader2008` implementation of CP-APR :cite:p:`ChKo12`.



Resources
========================================
* `Example Notebooks <https://github.com/lanl/pyCP_APR/tree/main/examples>`_
* `Example Tensors <https://github.com/lanl/pyCP_APR/tree/main/data/tensors>`_
* `Paper <https://ieeexplore.ieee.org/abstract/document/9280524>`_
* `Website <https://smart-tensors.lanl.gov>`_
* `Code <https://github.com/lanl/pyCP_APR>`_

Installation
========================================
**Option 1: Install using pip**

.. code-block:: shell

    pip install git+https://github.com/lanl/pyCP_APR.git

**Option 2: Install from source**

.. code-block:: shell

    git clone https://github.com/lanl/pyCP_APR.git
    cd pyCP_APR
    conda create --name pyCP_APR python=3.9
    source activate pyCP_APR
    pip install -e . # or <python setup.py install>

**Optional Tutorial for Examples:**
Jupyter Setup Tutorial for using the examples (`Link <https://www.maksimeren.com/post/conda-and-jupyter-setup-for-research/>`_)

Example Usage
========================================
.. code-block:: python

    from pyCP_APR import CP_APR
    from pyCP_APR.datasets import load_dataset

    # Load a sample tensor
    data = load_dataset(name="TOY")
    coords_train, nnz_train = data['train_coords'], data['train_count']
    coords_test, nnz_test = data['test_coords'], data['test_count']

    # CP-APR Object with PyTorch backend on a GPU. Transfer the latent factors back to Numpy arrays.
    model = CP_APR(n_iters=10,
                   random_state=42,
                   verbose=1,
                   method='torch',
                   device='gpu',
                   return_type='numpy')

    # Train a rank 45 tensor
    M = model.fit(coords=coords_train, values=nnz_train, rank=45)

    # Predict the scores over the trained tensor
    y_score = model.predict_scores(coords=coords_test, values=nnz_test)


How to Cite pyCP_APR?
========================================
.. code-block:: console

    @MISC{Eren2021pyCPAPR,
      author = {M. E. {Eren} and J. S. {Moore} and E. {Skau} and M. {Bhattarai} and G. {Chennupati} and B. S. {Alexandrov}},
      title = {pyCP\_APR},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      doi = {10.5281/zenodo.4840598},
      howpublished = {\url{https://github.com/lanl/pyCP\_APR}}
    }


    @INPROCEEDINGS{Eren2020ISI,
      author={M. E. {Eren} and J. S. {Moore} and B. S. {Alexandrov}},
      booktitle={2020 IEEE International Conference on Intelligence and Security Informatics (ISI)},
      title={Multi-Dimensional Anomalous Entity Detection via Poisson Tensor Factorization},
      year={2020},
      pages={1-6},
      doi={10.1109/ISI49825.2020.9280524}
    }


Authors
========================================
- `Maksim Ekin Eren <mailto:maksim@lanl.gov>`_: Advanced Research in Cyber Systems, Los Alamos National Laboratory
- `Juston S. Moore <mailto:jmoore01@lanl.gov>`_: Advanced Research in Cyber Systems, Los Alamos National Laboratory
- `Erik Skau <mailto:ewskau@lanl.gov>`_: Computer, Computational, and Statistical Sciences Division, Los Alamos National Laboratory
- `Manish Bhattarai <mailto:ceodspspectrum@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory
- `Gopinath Chennupati <mailto:cgnath.dr@gmail.com>`_: Computer, Computational, and Statistical Sciences Division, Los Alamos National Laboratory
- `Boian S. Alexandrov <mailto:boian@lanl.gov>`_: Theoretical Division, Los Alamos National Laboratory

Acknowledgments
========================================
We thank Austin Thresher for the valuable feedback on our software design.

Copyright Notice
========================================
© 2021. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

**LANL C Number: C21028**


License
========================================
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

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


Developer Test Suite
========================================
Developer test suites are located under ``tests/`` directory (located `here <https://github.com/lanl/pyCP_APR/tree/main/tests>`_).

Tests can be ran from this folder using ``python -m unittest *``.


References
========================================

.. bibliography:: refs.bib


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   CP_APR
   datasets
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
