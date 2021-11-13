"""
pyCP_APR.py is the Scikit-learn like API for interacting with the CP-APR algorithm. The **pyCP_APR.CP_APR** wraps
both the Numpy and PyTorch backend. **pyCP_APR.CP_APR** also includes API calls for anomaly detection utilities. Sparse tensor entries are scored by calculating p-values over the fitted model, where the lower p-value scores are an indicator for anomaly.\n

The fitted model (or factorized tensor, i.e. the KRUSKAL tensor M) during the training time describes the normal or the expected behavior which follows the Poisson distribution. We say that the entries of the tensor in the test set are drawn from the same distribution
as the factorized training tensor M. Using M, we calculate how likely the entries of the test tensor to occur given what was expected. This methodology was introduced by Eren et al. in [1].

Some code comments are borrowed from the original implementation of CP-APR in [2-5].


References
========================================
[1] M. E. Eren, J. S. Moore and B. S. Alexandrov, "Multi-Dimensional Anomalous Entity Detection via Poisson Tensor Factorization," 2020 IEEE International Conference on Intelligence and Security Informatics (ISI), 2020, pp. 1-6, doi: 10.1109/ISI49825.2020.9280524.\n
[2] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[3] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[4] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[5] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.\n

@author Maksim Ekin Eren
"""
import sys
import numpy as np

from .torch_backend.CP_APR_Torch import CP_APR_MU as CP_APR_MU_tr
from .numpy_backend.CP_APR import CP_APR_MU

from .applications.tensor_anomaly_detection import PoissonTensorAnomaly
from .applications.tensor_anomaly_detection_v2 import PoissonTensorAnomaly_v2 as PTA

class CP_APR():


    def __init__(self, **parameters):
        """
        Initilize the **pyCP_APR.CP_APR** class. **pyCP_APR.CP_APR** class is the wrapper for the CP-APR algorithm's Python implementation with
        both *Numpy* and *PyTorch* backend.

        Parameters
        ----------
        method : string, optional
            Specifies which backend to use when running CP-APR, and sets the model (i.e. **pyCP_APR.CP_APR.model**)
            accordingly.\n
            ``method='torch'`` or ``method='pytorch'`` will use PyTorch and enable GPU utilization.\n
            ``method='numpy'`` will use Numpy backend.\n
            Default is ``method='torch'``.
            
            .. warning::
            
                * ``method='torch'`` or ``method='pytorch'`` only supports sparse tensors in COO format.
                * ``method='numpy'`` supports both sparse (COO format) & dense tensors.
            
        epsilon : float, optional
            Prevents zero division. Default is 1e-10.
        kappa : float, optional
            Fix slackness level. Default is 1e-2.
        kappa_tol : float, optional
            Tolerance on slackness level. Default is 1e-10.
        max_inner_iters : int, optional
            Number of inner iterations per epoch. Default is 10.
        follow_M : bool, optional
            Saves M on each iteration if ``True``.
            The default is False.
        n_iters : int, optional
            Number of iterations during optimization or epoch. Default is 1000.
        print_inner_itn : int, optional
            Print every *n* inner iterations. Does not print if 0. Default is 0.
        verbose : int, optional
            Print every n epoch, or ``n_iters``. Does not print if 0. Default is 10.
        stoptime : float, optional
            Number of seconds before early stopping. Default is 1e6.
        tol : float, optional
             KKT violations tolerance. Default is 1e-4.
        random_state : int, optional
            Random seed for the initial M (KRUSKAL Tensor). Default is 42.
        device : string, optional
            Specifies CPU or GPU utilization for factorizing the tensor.\n
            ``device='cpu'`` to use PyTorch with CPU.\n
            ``device='gpu'`` to use PyTorch with GPU. Only if a CUDA device is available.\n
            Default is ``device='cpu'``.
            
            .. warning::
            
                Only used when ``method='torch'`` or ``method='pytorch'``.
                
        device_num : int or string, optional
            Which GPU to use to compute the KRUSKAL tensor M.\n
            Default is ``device_num=0``.
            
            .. warning::
                
                 Only used when ``method='torch'`` and ``device='gpu'``.

        return_type : string, optional
            The return type for the final KRUSKAL tensor M.\n
            ``return_type='torch'`` keep as torch tensors.\n
            ``return_type='numpy'`` convert to Numpy arrays and transfer the tensor back to CPU if GPU was used.\n
            Default is ``return_type='numpy'``.
            
            .. warning::
            
                 Only used when ``method='torch'`` or ``method='pytorch'``.
            
        dtype : string, optional
            Type to be used in torch tensors.\n
            Default is **'torch.DoubleTensor'** when ``device='cpu'``. Default is **'torch.cuda.DoubleTensor'** when ``device='gpu'``.\n
            
            .. warning::
            
                Used only when ``method='torch'`` or ``method='pytorch'``.
        
        .. note::
        
            **Example**

            Using the *PyTorch* backend on GPU 0:

            .. code-block:: python

                from pyCP_APR.pyCP_APR import CP_APR

                # CP-APR Object with PyTorch backend on a GPU. Transfer the latent factors back to Numpy arrays.
                cp_apr = CP_APR(n_iters=10, random_state=42, verbose=1, device='gpu', device_num=0, return_type='numpy')


            Using the *Numpy* backend:


            .. code-block:: python

                cp_apr = CP_APR(n_iters=10, random_state=42, verbose=1, method='numpy')


        """
        
        # Compute method
        if 'method' not in parameters:
            self.method = 'torch'
        else:
            self.method = parameters['method']
            del parameters['method']

            allowed_methods = ['torch', 'pytorch', 'numpy']
            assert self.method in allowed_methods, "Unknown method. Please choose from: %s" % str(','.join(allowed_methods))

        # CP-APR PyTorch
        if self.method == 'torch' or self.method == 'pytorch':
            self.model = CP_APR_MU_tr(**parameters)
        # CP-APR Numpy
        elif self.method == 'numpy':
            self.model = CP_APR_MU(**parameters)

        # Save the results
        self.M = None
        self.score = None
        self.R = -1
        self.prediction = None
        self.PTA = None

    def get_params(self):
        """
        The function call that returns the *model* parameters in a dictionary where a key is the *model* variable name
        and the value is its current value. *model* is the backend used during factorization (i.e. **pyCP_APR.torch_cp.CP_APR_Torch** or 
        **pyCP_APR.numpy_cp.CP_APR**).\n


        .. note::
        
            Model parameters can also be accessed with a call directly to the *model* (i.e. **pyCP_APR.CP_APR.model**).

        
        .. note::
        
            **Example**


            .. code-block:: python

                from pyCP_APR import CP_APR
                cp_apr = CP_APR(n_iters=10)
                cp_apr.get_params()


            .. code-block:: console

                {
                 'verbose': 10,
                 'print_inner_itn': 0,
                 'start_time': -1,
                 'final_iter': -1,
                 'dtype': 'torch.DoubleTensor',
                 'device': 'cpu',
                 'device_num': '0',
                 'return_type': 'numpy',
                 'X': None,
                 'M': None,
                 'tol': 0.0001,
                 'stoptime': 1000000.0,
                 'exec_time': -1,
                 'n_iters': 10,
                 'max_inner_iters': 10,
                 'random_state': 42,
                 'kappa': 0.01,
                 'kappa_tol': 1e-10,
                 'kktViolations': tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
                 'nInnerIters': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 'times': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 'logLikelihoods': tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                 'epsilon': tensor(1.0000e-10),
                 'obj': 0
                 }


            The model parameters of a fitted model may look like following:


            .. code-block:: console

                {
                 'verbose': 1,
                 'print_inner_itn': 0,
                 'start_time': 1621842138.9178197,
                 'final_iter': 10,
                 'dtype': 'torch.cuda.DoubleTensor',
                 'device': device(type='cuda', index=0),
                 'device_num': '0',
                 'return_type': 'numpy',
                 'X': <pyCP_APR.torch_cp.sptensor_Torch.SP_TENSOR at 0x7f759dde0eb0>,
                 'M': <pyCP_APR.torch_cp.ktensor_Torch.K_TENSOR at 0x7f759dde0c40>,
                 'tol': 0.0001,
                 'stoptime': 1000000.0,
                 'exec_time': 0.41489696502685547,
                 'n_iters': 10,
                 'max_inner_iters': 10,
                 'random_state': 42,
                 'kappa': 0.01,
                 'kappa_tol': 1e-10,
                 'kktViolations': array([  0.90469806, 170.76687748, 439.2454979 , 171.40815408,
                         32.90351131,  49.3060232 , 112.32568973, 156.01606934,
                         30.25656979,  24.61207396]),
                 'nInnerIters': array([38., 34., 34., 33., 33., 33., 33., 33., 33., 33.]),
                 'times': array([0.05231047, 0.09365845, 0.14037013, 0.18132734, 0.22193289,
                        0.26159644, 0.29836512, 0.33511233, 0.37501359, 0.41115212]),
                 'logLikelihoods': array([9.59298322e+12, 1.02443325e+13, 1.03623011e+13, 1.04159710e+13,
                        1.04551136e+13, 1.04828857e+13, 1.05091620e+13, 1.05297763e+13,
                        1.05492323e+13, 1.05635448e+13]),
                 'epsilon': array(1.e-10),
                 'obj': 10563544752580.432
                 }


        """
        return vars(self.model)

    def get_score(self):
        """
        The function call that returns the objective value of the model.\n
        Model is **pyCP_APR.CP_APR.model.obj**.\n
        If an ensemble of tensors is trained (see **pyCP_APR.CP_APR.fit()**), a list of scores is provided instead.\n
        Note that the model has to be already fit.

        Returns
        -------
        score : float
            Model fit score.

        """
        assert self.M is not None, "Fit the tensor to aquire the latent factors first."
        return self.score

    def get_prediction(self):
        """
        The function call that returns the predictions.\n
        The model must be already fitted using **pyCP_APR.CP_APR.fit()**,
        and links must be already predicted using **pyCP_APR.CP_APR.predict_scores()**.\n
        
        .. warning::
        
            The prediction is returned as dictionary with the keys 'objective' and 'lambda'.\n
            The 'objective' key depends on the ``objective`` parameter used when predicting the scores in **pyCP_APR.CP_APR.predict_scores()**.

        Returns
        -------
        predictions : dict
            The prediction contents are based on the objective parameter used during **pyCP_APR.CP_APR.predict_scores()**.\n
            If ``objective='p_value'``, returns ``{'p_value': array, 'lambda': array}``.\n
            If ``objective='log_likelihood'``, returns ``{'log_likelihood': array, 'lambda': array}``.

        
        .. note::
        
            **Example**


            .. code-block:: python

                cp_apr.get_prediction()


            .. code-block:: console

                {
                 'p_value': array([1., 1., 1., ..., 1., 1., 1.]),
                 'lambda': array([ 3796.43165171,  2315.69440274,  1001.22377495, ..., 290.35037293,  2952.72557334, 30309.82134089])
                }


        """
        assert self.prediction is not None, "Perform the predictions over the latent factors first."
        return self.prediction

    def set_params(self, **parameters):
        """
        Sets the model parameters.\n
        Here the model is **pyCP_APR.CP_APR.model**.

        Parameters
        ----------
        parameters : dict
            All model parameters. See **pyCP_APR.CP_APR** class initilization parameters.

        Returns
        -------
        self : object
            self is returned.

        """

        for parameter, value in parameters.items():
            setattr(self.model, parameter, value)
        return self

    def predict_probas(self, y, axis_map=None):
        """
        Calculates the probabilities of the test tensor entries.\n
        Then returns the prediction scores with ROC-AUC and PR-AUC metrics.\n
        Fusion is performed on the dimensions indicated by ``axis_map``.
        Dimension fusion uses the Harmonic Mean.\n
        **This function is in beta stage.**

        
        .. note::
            Anomaly detection can be performed either using **predict_scores()**, or using **transform()** and **predict_probas()**. 
            While using **transform()** and **predict_probas()** yields faster computation time and more established dimension fusion results, **predict_scores()** provides wider range of features for anomaly detection.


        .. warning:: 
            In order to use the **predict_probas()** function, below has to be done first:

            1. Tensor has to be factorized using **fit()** function first to extract the KRUSKAL tensor M.
            2. After **fit()**, Poisson lambda values for the test tensor has to be calculated using the **transform()** function.


        Parameters
        ----------
        y : list or array
            Labels for each entry in the sparse test tensor.
        axis_map : OrderedDict, optional
            If fusing dimensions, list of dimension numbers can be passed as *OrderedDict* to identify which dimensions to fuse.\n
            The tuples in the ordered dictionary map will have 2 entries, where the first entry is the dimension name in string, and the
            second entry is the dimension number(s) in list.
            The default is ``axis_map=None``.

        Returns
        -------
        prediction scores : dict or Pandas DataFrame
            If not fusing the dimensions over the transformed tensor, returns a dictionary with prediction scores.\n
            ``{"roc_auc":float, "pr_auc":float}``\n
            If dimensions are being fused using the ``axis_map`` parameter, returns a *Pandas DataFrame* with the prediction
            scores. The columns for the returned DataFrame in this case are 'fusion', 'method', 'metric', and 'score'.\n
            
            .. note::
            
                For a four dimensional tensor with dimension names *U x S x D x s*, ``axis_map`` to fuse to first dimension, and first and second 
                dimensions would be: ``axis_map=OrderedDict((('U', [0]), ('US', [0, 1])))``. Here ``'U'`` and ``'US'`` are the dimension names,
                and ``[0]`` and ``[0,1]`` are the dimension numbers respectively.\n
                Another example is illustrated below.

            
                .. code-block:: python

                    from collections import OrderedDict
                    axis_map = OrderedDict((('U', [0]), ('S', [1]), ('D', [2]), ('US', [0, 1]), ('UD', [0, 2]), ('SD', [1, 2])))

        """
        assert self.M is not None, "Fit the tensor to aquire the latent factors first."
        assert self.PTA is not None, "Transform the indicies in the latent factors first."
        if axis_map != None:
            return self.PTA.get_dimension_fusion_scores(axis_map, y)
        else:
            return self.PTA.get_link_prediction_scores(y)

    def transform(self, indicies, ensemble_significance = []):
        """
        Given the sparse test tensor entries (or indicies, i.e. coordinates of non-zero values), calculate the Poisson lambda values.\n
        The Poission lambda value can be used to identify how likely was that coordinate in the test tensor to occur given
        what we have learned during the training time.\n
        
        .. note::
        
            * If 2 tensors are weighted during the calculation of lambdas, the weight of each can be specified using ``ensemble_significance``. See **pyCP_APR.CP_APR.fit()** for factorazing ensemble of tensor ranks. For instance, if ``ensemble_significance=[0.1, 0.9]``, ``lambda = (0.1 x lambda_1) + (0.9 x lambda_2)``.
            * Anomaly detection can be performed either using **predict_scores()**, or using **transform()** and **predict_probas()**. While using **transform()** and **predict_probas()** yields faster computation time and more established dimension fusion results, **predict_scores()** provides wider range of features for anomaly detection.
        
        .. warning:: 
            To use the **transform()** function, the model has to be **fit()** first.


        Parameters
        ----------
        indicies : list or array
            List of non-zero coordinates in the test tensor.
        ensemble_significance : list, optional
            The weight of each tensor during lambda calculations.\n
            Ensemble significance is automatically applied if multiple tensors are fitted.\n
            If multiple tensors are fitted, the default is ``ensemble_significance=[0.1, 0.9]``.\n
            If single tensor is fitted, the default is ``ensemble_significance=[1]``.

        Returns
        -------
        lambdas : array
            List of lambda values for each of the non-zero coordinates.
        
        
        .. note::
        
            **Example**


            .. code-block:: python

                from pyCP_APR.datasets import load_dataset
                from pyCP_APR import CP_APR
                from collections import OrderedDict
                import numpy as np

                data = load_dataset(name = "TOY")

                # Training set
                coords_train = data['train_coords']
                nnz_train = data['train_count']

                # Test set
                coords_test = data['test_coords']
                nnz_test = data['test_count']

                # CP-APR model
                cp_apr = CP_APR(n_iters=10, random_state=42, verbose=1, method='numpy')

                # factorize X for ranks 1 and 4
                M = cp_apr.fit(coords=coords_train, values=nnz_train, rank=[1,4])

                # get the lambdas
                # returned lambdas are weighted values for rank 1 and rank 4 tensor decomposition Ms
                lambdas = cp_apr.transform(coords_test)


        """
        assert self.M is not None, "Fit the tensor to aquire the latent factors first."

        if len(ensemble_significance) == 0:
            if isinstance(self.R, (list, np.ndarray)):
                ensemble_significance = [0.1, 0.9]
            else:
                ensemble_significance = [1]

        self.PTA = PTA(self.M, indicies, ensemble_significance)
        return self.PTA.get_lambdas()

    def fit(self, **parameters):
        """
        Takes the decomposition of sparse or dense tensor X and returns the KRUSKAL tensor M.\n
        Here M is  latent factors and the weight of each R (rank) component.\n
        If a list of ranks is passed, factorize the tensor for each 2 of the ranks.\n
        The factorized 2 tensors M in this case will follow the weighted lambda calculations during prediction.


        Parameters
        ----------
        tensor : PyTorch.sparse or Numpy array
            Original dense or sparse tensor X.\n
            Can be used when ``Type='sptensor'``. In this case, ``Tensor`` needs to be a *PyTorch Sparse* tensor format.\n
            Or use with ``Type='tensor'`` and pass ``Tensor`` as a dense Numpy array.

            .. warning::

                Note that PyTorch backend only supports ``Type='sptensor'``.

        coords : Numpy array
            Array of non-zero coordinates for sparse tensor X. COO format.\n
            Each entry in this array is a coordinate of a non-zero value in the original tensor X.

            .. warning::
            
                * Used when ``Type='sptensor'``, and when ``tensor`` parameter is not passed.
                * ``len(coords)`` is number of total entiries in X, and ``len(coords[0])`` should give the number of dimensions X has.

        values : Numpy array
            List of non-zero values corresponding to each list of non-zero coordinates (``coords``).
            Array of non-zero tensor entries. COO format.

            .. warning::

                * Used when ``Type='sptensor'`` and ``tensor`` parameter is not passed.
                * Length of ``values`` must match the length of ``coords``.

        rank : int or list
            Tensor rank, or list of ranks for two tensors.\n
            Tensor rank determines the number of components.\n
            List of ranks will allow using weighted prediction between the two latent factors in KRUSKAL tensor M.\n
            Pass a single integer or list of length two.\n
            The default is ``rank=2``.
        Minit : string or dictionary of latent factors
            Initial value of latent factors.\n
            If ``Minit='random'``, initial factors are drawn randomly from uniform distribution between 0 and 1.\n
            Else, pass a dictionary where the key is the mode number and value is array size ``d x r``
            where ``d`` is the number of elements on the dimension and ``r`` is the rank.\n
            The default is ``Minit='random'``.
            
            .. note::
            
                Example on creating initial M for 3 dimensional tensor shaped *5x5x5* for rank 4 decomposition:

                .. code-block:: python

                    import numpy as np

                    num_dimensions = 3
                    tensor_shape = [5,5,5]
                    rank = 4
                    M_init = {"Factors":{}, "Weights":[1,1,1]}
                    for d in range(num_dimensions):
                            M_init["Factors"][str(d)] = np.random.uniform(low=0, high=1, size=(tensor_shape[d], rank))
                    M_init["Factors"]
                    
                .. code-block:: console
                
                    {
                     '0': array([[0.821161  , 0.419537  , 0.62692165, 0.06294969],
                            [0.02032657, 0.88625546, 0.74128504, 0.71855629],
                            [0.70760879, 0.83813636, 0.35128158, 0.94442011],
                            [0.35780608, 0.83703369, 0.84602297, 0.93760842],
                            [0.00746915, 0.05974905, 0.49097518, 0.60615737]]),
                     '1': array([[0.61902526, 0.78453503, 0.05596952, 0.69149084],
                            [0.56300552, 0.82418509, 0.04278352, 0.25716303],
                            [0.66221183, 0.13888761, 0.92502242, 0.57817265],
                            [0.31738958, 0.87061048, 0.64170398, 0.62236073],
                            [0.9110603 , 0.5133135 , 0.89232955, 0.09881775]]),
                     '2': array([[0.0580065 , 0.82367217, 0.07616138, 0.93873983],
                            [0.89247679, 0.41388867, 0.82089524, 0.10293565],
                            [0.13540868, 0.09809637, 0.10844113, 0.90405324],
                            [0.91167498, 0.67068632, 0.51705956, 0.82211517],
                            [0.80942828, 0.08450466, 0.6306868 , 0.78132797]])
                    }

        Type : string
            Type of tensor (i.e. sparse or dense).\n
            Use ``Type='sptensor'`` for sparse, and ``Type='tensor'`` for dense tensors.\n
            'sptensor' can be used with ``method='torch'``, and ``method='numpy'``.\n
            If 'sptensor' used, pass the list of non-zero coordinates using the ``coords`` parameter
            and the corresponding list of non-zero elements with ``values`` parameter. This is the COO representation of X.\n
            'sptensor' can also be used with the *PyTorch Sparse* format. 
            In that case, pass the tensor X that is in *torch.sparse* format using the ``tensor`` parameter.\n
            'tensor' can be used with ``method='numpy'``. In this case, pass the tensor X using the ``tensor`` parameter.\n
            The default is ``Type='sptensor'``.


        Returns
        -------
        KRUSKAL tensor M : dict
            KRUSKAL tensor M is returned in dict format.\n
            The latent factors can be found with the key 'Factors'.\n
            The weight of each component can be found with the key 'Weights'.


        .. note::
        
            **Example**

            Sparse tensor X in COO format decomposed using a GPU in the below example:

            .. code-block:: python

                from pyCP_APR.datasets import load_dataset
                from pyCP_APR import CP_APR
                from collections import OrderedDict
                import numpy as np

                data = load_dataset(name = "TOY")

                # Training set
                coords_train = data['train_coords']
                nnz_train = data['train_count']

                # Test set
                coords_test = data['test_coords']
                nnz_test = data['test_count']

                # CP-APR model
                cp_apr = CP_APR(n_iters=10, random_state=42, verbose=1, 
                    method='torch',
                    device='gpu', 
                    device_num=0
                   )

                # factorize the tensor for ranks 1 and 4
                M = cp_apr.fit(coords=coords_train, values=nnz_train, rank=[1,4])


            Above example takes the tensor decomposition of X for ranks 1 and 4. Below is an example showing a single rank decomposition:


            .. code-block:: python

                M = cp_apr.fit(coords=coords_train, values=nnz_train, rank=4)


            An example of factorized X, i.e. M (KRUSKAL tensor). Below example M is rank 2, and has 3 dimensions:


            .. code-block:: console

                {
                   'Factors': 
                   {
                      '0': 
                         array([[5.88838457e-51, 2.13058370e-01],
                         [3.23364716e-04, 1.34610100e-01],
                         [2.05013230e-01, 7.12928005e-37],
                         [1.48424405e-01, 0.00000000e+00],
                         [9.76200219e-02, 1.48477484e-01],
                         [2.51566211e-02, 2.06908903e-01],
                         [1.43573934e-01, 3.34319439e-88],
                         [2.61925420e-01, 4.76257924e-33],
                         [9.37106506e-02, 1.87295857e-01],
                         [2.42523537e-02, 1.09649287e-01]]),
                      '1': 
                         array([[2.31775360e-241, 5.03672967e-002],
                         [7.79309622e-002, 1.00144467e-137],
                         [0.00000000e+000, 7.84481789e-002],
                         [1.23105143e-001, 9.77480876e-002],
                         [3.30736653e-002, 5.64828345e-002],
                         [1.56285154e-078, 9.36029407e-003],
                         [4.85047483e-002, 0.00000000e+000],
                         [3.10430389e-002, 0.00000000e+000],
                         [2.39290092e-002, 3.29838934e-002],
                         [0.00000000e+000, 0.00000000e+000],
                         [6.75832826e-002, 0.00000000e+000]]),
                      '2': 
                         array([[2.71626813e-002, 0.00000000e+000],
                         [1.68530286e-003, 4.18040234e-002],
                         [0.00000000e+000, 2.00577503e-002],
                         [0.00000000e+000, 5.34873341e-002],
                         [0.00000000e+000, 2.89723060e-002],
                         [4.00972915e-002, 4.85187813e-161],
                         [4.49477703e-002, 0.00000000e+000],
                         [1.00243229e-002, 0.00000000e+000],
                         [0.00000000e+000, 3.69954061e-002],
                         [2.29589330e-002, 1.33718335e-002],
                         [3.23365254e-002, 0.00000000e+000]])},
                  'Weights': array([3092.47820339, 2243.52179661])
                }


            Below is an example of how *torch.sparse* format can be used as the tensor X:


            .. code-block:: python

                import torch

                i = torch.LongTensor([[0, 1, 1], [2, 0, 2], [2, 0, 1]])

                v = torch.FloatTensor([3, 4, 5])
                X = torch.sparse.FloatTensor(i, v, torch.Size([4,4,4]))


            .. code-block:: python

                from pyCP_APR import CP_APR

                cp_apr = CP_APR(n_iters=100, verbose=10, device='gpu')
                result = cp_apr.fit(tensor=X, rank=30)


            .. code-block:: console

                Using TITAN RTX
                CP-APR (MU):
                Iter=1, Inner Iter=30, KKT Violation=0.425532, obj=4.887921, nViolations=0
                Exiting because all subproblems reached KKT tol.
                ===========================================
                 Final log-likelihood = 4.888204
                 Final least squares fit = 0.999995
                 Final KKT violation = 0.000007
                 Total inner iterations = 37
                 Total execution time = 0.2447 seconds
                Converting the latent factors to Numpy arrays.


            Below is an example on using a dense Numpy array as tensor X:


            .. code-block:: python

                import numpy as np
                from pyCP_APR import CP_APR

                # X has the shape 10 x 30 x 40
                X = np.arange(1, 12001).reshape([10,30,40])

                cp_apr = CP_APR(n_iters=100, verbose=10, method='numpy')
                result = cp_apr.fit(tensor=X, Type='tensor', rank=2)


            .. code-block:: console

                CP-APR (MU):
                Iter=1, Inner Iter=30, KKT Violation=0.244501, obj=534739600.517348, nViolations=0
                Exiting because all subproblems reached KKT tol.
                ===========================================
                 Final log-likelihood = 534841753.347965
                 Final least squares fit = 0.971281
                 Final KKT violation = 0.000091
                 Total inner iterations = 161
                 Total execution time = 1.1126 seconds


            .. code-block:: python

                result.keys()


            .. code-block:: console

                dict_keys(['Factors', 'Weights'])


            .. code-block:: python

                M = result['Factors']
                Gamma = result['Weights']

                M_0 = M['0']
                Gamma_0 = Gamma[0]

                print('Component 0:', M_0, 'Gamma 0:', Gamma_0)


            .. code-block:: console

                Component 0:
                [[0.01002107 0.0099889 ]
                [0.03001639 0.02999136]
                [0.05001171 0.04999383]
                [0.07000709 0.0699962 ]
                [0.09000233 0.08999878]
                [0.10999784 0.11000099]
                [0.12999289 0.13000382]
                [0.14998825 0.15000623]
                [0.16998359 0.17000867]
                [0.18997884 0.19001122]] 
                Gamma 0: 41633867.33685632


        """

        # Train M with multiple ranks
        if 'rank' in parameters and isinstance(parameters['rank'], (list, np.ndarray)):

            Rank = parameters['rank']

            # Currently only supports 2 at a time for weighted prediction
            assert len(Rank) == 2, "Secify list of two ranks such as [2,4], or a single intiger rank."

            self.M = list()
            self.score = list()
            self.R = list()
            for r in Rank:
                parameters['rank'] = r
                self.M.append(self.model.train(**parameters))
                self.score.append(self.model.obj)
                self.R.append(r)

        else:
            self.M = self.model.train(**parameters)
            self.score = self.model.obj

            # Save the rank
            self.R = self.model.M.Rank

        return self.M

    def predict_scores(self, **parameters):
        """
        The function call that can be used for classification of anomalies after fitting the tensor.
        The model will use the trained latent factors to generate the Poisson lambda scores corresponding to the given
        new coordinate.\n
        These lambda values are then used to calculate the p-values for classification of the entries.\n
        The lower p-value here is an indicator of an anomaly.\n
        Since the learned or expected behaviour during the training time is represented by the KRUSKAL tensor M,
        we can calculate how likely a new index to occur in M (i.e. M represents the average number of expected events for each coordinate).
        If two tensors trained during fitting, the prediction will weight the lambdas before calculating the p-values.


        .. note::
        
            * We find that using ensemble of tensors during prediction significantly reduces the false positive rates for anomaly detection as shown in [2].
            * Anomaly detection can be performed either using **predict_scores()**, or using **transform()** and **predict_probas()**. While using **transform()** and **predict_probas()** yields faster computation time and more established dimension fusion results, **predict_scores()** provides wider range of features for anomaly detection. 


        .. warning:: 

            * To use **predict_scores()**, **fit()** the model first.


        Parameters
        ----------
        coords : array
            Coordinates of the non-zero values.\n
        values : list
            List of non-zero values.\n
            Length must match the ``coords`` parameter length.\n
            Example binary links: ``array([1, 1, 1])``.
        from_matlab : boolean, optional
            If the dataset used in MATLAB as well, indices may start at 1 instead of 0. \n
            This parameter can be used to subtract 1 from the indices.\n
            The default is False.
        objective : string, optional
            ``objective='p_value'`` calculates the Poisson p-value.\n
            Fusion ``objective`` options: 'p_value_fusion_harmonic', 'p_value_fusion_harmonic_observed', 'p_value_fusion_chi2',
            'p_value_fusion_chi2_observed', 'p_value_fusion_arithmetic'.\n
            If fusion is being used, specify the list of dimensions that are being targeted
            via the ``p_value_fusion_index`` parameter.\n
            Calculate log_likelihood of observing the link with ``objective='log_likelihood'``.\n
            The default is ``objective='p_value'``.
        ensemble_significance : list of length two, optional
            Weight of each tensor, if two is trained. Two is trained when ``rank=[r1,r2]`` during **pyCP_APR.CP_APR.fit()**
            where *r1* and *r2* are intiger ranks.
            The default is ``ensemble_significance=[0.1, 0.9]``.
        p_value_fusion_index : list, optional
            Fuses down to the target dimensions.\n
            List should contain the index of the dimensions to fuse.
            The default is ``p_value_fusion_index=[0]``.

            .. warning::
            
                Only used if fusion objective is being used.

        ignore_dimensions_indx : list, optional
            If used, the dimension numbers in the list will be ignored during the calculation of the lambdas.

        Returns
        -------
        predictions : array
            Returns the prediction objective.\n
            For instance, if parameter was ``objective='p_value'``, array of p-values are returned for each entry in the test tensor.


        .. note::

            **Example**

            Sample coordinate and value pair for a four dimensional tensor with 3 entries:

            .. code-block:: python

                # coordinates of 3 entries of 4 dimensional tensor
                coords = array([[    0,   961,     0,     0],
                                [    0,   961,  1742,     0],
                                [    0,   961,  2588,     0]])
                values = [1,2,1]


            Extracting the p-values from the test tensor:

            .. code-block:: python

                from pyCP_APR.datasets import load_dataset
                from pyCP_APR import CP_APR
                from collections import OrderedDict
                import numpy as np

                data = load_dataset(name = "TOY")

                # Training set
                coords_train = data['train_coords']
                nnz_train = data['train_count']

                # Test set
                coords_test = data['test_coords']
                nnz_test = data['test_count']

                # CP-APR model
                cp_apr = CP_APR(n_iters=10, random_state=42, verbose=1, method='torch', device='gpu', device_num=0)

                # factorize the tensor for ranks 1 and 4.
                M = cp_apr.fit(coords=coords_train, values=nnz_train, rank=[1,4])

                # calculate the p-values for the entries in the test set
                p_values = cp_apr.predict_scores(coords=coords_test, values=nnz_test)


            These p-values are also saved in the class variable, and can be found as follows:


            .. code-block:: python

                scores = list(cp_apr.prediction['p_value'])


        """

        assert self.M is not None, "Fit the tensor to aquire the latent factors first."

        # Prediction objective
        if 'objective' not in parameters:
            objective = 'p_value'
        else:
            objective = parameters['objective']
            del parameters['objective']

        # Lambda calculation method
        if 'ensemble_significance' not in parameters:
            ensemble_significance = [0.1, 0.9]
        else:
            ensemble_significance = parameters['ensemble_significance']
            del parameters['ensemble_significance']

        # If fusion is being performed
        if 'p_value_fusion_index' not in parameters:
            p_value_fusion_index = [0]
        else:
            p_value_fusion_index = parameters['p_value_fusion_index']
            del parameters['p_value_fusion_index']

        # if we are ignoring a dimension when calculating the lambdas
        if 'ignore_dimensions_indx' not in parameters:
            ignore_dimensions_indx = []
        else:
            ignore_dimensions_indx = parameters['ignore_dimensions_indx']
            del parameters['ignore_dimensions_indx']


        # If weighting two tensors during the calculation
        if isinstance(self.R, (list, np.ndarray)):
            PTA = PoissonTensorAnomaly(dimensions=self.M[0]['Factors'],
                                       weights=self.M[0]['Weights'][0],
                                       objective=objective,
                                       lambda_method='ensemble',
                                       ensemble_dimensions=self.M[1]['Factors'],
                                       ensemble_weights=self.M[1]['Weights'],
                                       ensemble_significance=ensemble_significance,
                                       ignore_dimensions_indx=ignore_dimensions_indx
                                      )
        # Single tensor calculation
        else:
            PTA = PoissonTensorAnomaly(dimensions=self.M['Factors'],
                                       weights=self.M['Weights'],
                                       objective=objective,
                                       lambda_method='single_tensor',
                                       ignore_dimensions_indx=ignore_dimensions_indx
                                      )

        self.prediction = PTA.predict(**parameters)

        return self.prediction[str(objective)]
