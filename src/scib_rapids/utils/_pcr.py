import cupy as cp
import numpy as np
import pandas as pd

from scib_rapids._types import NdArray

from ._pca import pca
from ._utils import get_ndarray, one_hot


def principal_component_regression(
    X: NdArray,
    covariate: NdArray,
    categorical: bool = False,
    n_components: int | None = None,
) -> float:
    """Principal component regression (PCR) using CuPy.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    covariate
        Array of shape (n_cells,) or (n_cells, 1) representing batch/covariate values.
    categorical
        If True, batch will be treated as categorical and one-hot encoded.
    n_components
        Number of components to compute. If None, all components are used.

    Returns
    -------
    pcr: float
    """
    if len(X.shape) != 2:
        raise ValueError("Dimension mismatch: X must be 2-dimensional.")
    if X.shape[0] != covariate.shape[0]:
        raise ValueError("Dimension mismatch: X and batch must have the same number of samples.")
    if categorical:
        covariate = np.asarray(pd.Categorical(covariate).codes)
    else:
        covariate = np.asarray(covariate)

    covariate_gpu = one_hot(covariate) if categorical else cp.asarray(covariate.reshape((covariate.shape[0], 1)), dtype=cp.float32)

    pca_results = pca(X, n_components=n_components)

    # Center inputs for no intercept
    covariate_gpu = covariate_gpu - cp.mean(covariate_gpu, axis=0)

    X_pca = cp.asarray(pca_results.coordinates, dtype=cp.float32)
    var = cp.asarray(pca_results.variance, dtype=cp.float32)

    # lstsq
    residual_sum = cp.linalg.lstsq(covariate_gpu, X_pca, rcond=None)[1]
    total_sum = cp.sum((X_pca - cp.mean(X_pca, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = cp.maximum(0, 1 - residual_sum / total_sum)

    pcr = cp.dot(cp.ravel(r2), var) / cp.sum(var)
    return float(pcr.item())
