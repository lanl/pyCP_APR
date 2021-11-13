def redistribute(M, mode):
    """
    This function distributes the weights to a specified dimension or mode.\n

    Parameters
    ----------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.
    mode : int
        Dimension number.
        
    Returns
    -------
    M : object
        KRUSKAL tensor class. ktensor.K_TENSOR.

    """
    for r in range(M.Rank):
        M.Factors[str(mode)][:, r] *= M.Weights[r]
        M.Weights[r] = 1

    return M