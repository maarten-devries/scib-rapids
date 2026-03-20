from dataclasses import dataclass

import cupy as cp
import numpy as np

from scib_rapids._types import NdArray

from ._utils import get_ndarray


@dataclass
class _SVDResult:
    u: NdArray
    s: NdArray
    v: NdArray


@dataclass
class _PCAResult:
    coordinates: NdArray
    components: NdArray
    variance: NdArray
    variance_ratio: NdArray
    svd: _SVDResult | None = None


def _svd_flip(u: cp.ndarray, v: cp.ndarray, u_based_decision: bool = True):
    """Sign correction to ensure deterministic output from SVD."""
    if u_based_decision:
        max_abs_cols = cp.argmax(cp.abs(u), axis=0)
        signs = cp.sign(u[max_abs_cols, cp.arange(u.shape[1])])
    else:
        max_abs_rows = cp.argmax(cp.abs(v), axis=1)
        signs = cp.sign(v[cp.arange(v.shape[0]), max_abs_rows])
    u_ = u * signs
    v_ = v * signs[:, None]
    return u_, v_


def pca(
    X: NdArray,
    n_components: int | None = None,
    return_svd: bool = False,
) -> _PCAResult:
    """Principal component analysis (PCA) using CuPy.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    n_components
        Number of components to keep. If None, all components are kept.
    return_svd
        If True, also return the results from SVD.

    Returns
    -------
    results: _PCAResult
    """
    max_components = min(X.shape)
    if n_components and n_components > max_components:
        raise ValueError(f"n_components = {n_components} must be <= min(n_cells, n_features) = {max_components}")
    n_components = n_components or max_components

    X_gpu = cp.asarray(X, dtype=cp.float32)
    X_centered = X_gpu - cp.mean(X_gpu, axis=0)

    u, s, v = cp.linalg.svd(X_centered, full_matrices=False)
    u, v = _svd_flip(u, v)

    variance = (s**2) / (X_gpu.shape[0] - 1)
    total_variance = cp.sum(variance)
    variance_ratio = variance / total_variance

    # Select n_components
    coordinates = u[:, :n_components] * s[:n_components]
    components = v[:n_components]
    variance_ = variance[:n_components]
    variance_ratio_ = variance_ratio[:n_components]

    results = _PCAResult(
        coordinates=get_ndarray(coordinates),
        components=get_ndarray(components),
        variance=get_ndarray(variance_),
        variance_ratio=get_ndarray(variance_ratio_),
        svd=_SVDResult(u=get_ndarray(u), s=get_ndarray(s), v=get_ndarray(v)) if return_svd else None,
    )
    return results
