"""
datasets.py is used to load the example tensors.

@author Maksim Ekin Eren
"""
import numpy as np
import pandas as pd
import os
import sys
import requests

def list_datasets():
    """
    This function returns the list of tensor names that are available to load.\n
    If the listing is requested for the first time, a new directory for the datasets is created.

    Returns
    -------
    datasets : list
        List of tensor names that are available to load.


    .. note::

        **Example**


        .. code-block:: python

            from pyCP_APR.datasets import list_datasets

            list_datasets()


        .. code-block:: console

            ['TOY']

    """
    files = _get_file_paths()["files"]

    datasets = list()
    for file in files:
        if ".npz" in file:
            datasets.append(file.split(".")[0])

    return datasets


def load_dataset(name="TOY"):
    """
    Loads the tensor specified by its name.
    
    
    .. warning::
    
        If a dataset is requested for the first time, it gets downloaded from GitHub.


    Parameters
    ----------
    name : string, optional
        The name of the tensor to load. The default is ``name="TOY"``.

    Returns
    -------
    data : Numpy NPZ
        Tensor contents compressed in Numpy NPZ  format.
        
    

    
    .. note::
    
        **Example**


        .. code-block:: python

            from pyCP_APR.datasets import load_dataset

            # Load a sample authentication training and test tensors along with the labels
            data = load_dataset(name="TOY")
            coords_train, nnz_train = data['train_coords'], data['train_count']
            coords_test, nnz_test = data['test_coords'], data['test_count']


        Available tensor data can be listed as follows:


        .. code-block:: python

            data = load_dataset(name = "TOY")
            list(data)


        .. code-block:: console

            ['train_coords',
             'train_count',
             'test_coords',
             'test_count']

    """
    datasets = _get_file_paths(name+".npz")

    if name + str(".npz") not in datasets["files"]:
        sys.exit("Dataset is not found! Available datasets are: " + ", ".join(list_datasets()))

    return np.load(datasets["path"] + name + str(".npz"), allow_pickle=True)


def _get_file_paths(name=""):
    """
    Helper function to extract the absolute path to the dataset, and the list of files in the data folder.\n
    
    .. warning::
    
        If the directory for the data are not available, i.e. when the data is requested for the same time, 
        a new directory for the data is created.\n
        If a dataset is requested for the first time, this helper function downloads the data from the GitHub
        repository of pyCP_APR.

    Parameters
    ----------
    name : string
        The name of the dataset to download if it does not exist.\n
        The default is "".

    Returns
    -------
    dataset information : dict
        ``{"path":string, "files":list}``.

    """
    download = False
    dirname = os.path.dirname(__file__)
    files_dirname = os.path.join(dirname, 'data/')
    
    try:
        # attempt to read the data directory
        files = os.listdir(files_dirname)
    
    except FileNotFoundError:
        # first time calling, create the datasets directory
        os.mkdir(files_dirname)
        print("Created:", files_dirname)
        files = os.listdir(files_dirname)
       
    # if not only listing the datasets, and if dataset is not downloaded yet
    if name != "" and name not in files:
        download = True
        
    # if only listing the datasets and non is downloaded
    elif name == "" and len(files) == 0:
        print("No datasets are downloaded.")
        print("See https://github.com/lanl/pyCP_APR/tree/main/data/tensors for available datasets.")
        print("Example dataset name: \"TOY\".")
        print("Datasets will be downloaded when pyCP_APR.datasets.load_dataset(name=\"TOY\") is called first time.")
    
    # if we need to download the dataset
    if download:
        print("Downloading the dataset:", name)

        url = "https://github.com/lanl/pyCP_APR/raw/main/data/tensors/" + name
        r = requests.get(url, allow_redirects=True)
        open(files_dirname + name, 'wb').write(r.content)
        files = os.listdir(files_dirname)
        

    return {"path":files_dirname, "files":files}
