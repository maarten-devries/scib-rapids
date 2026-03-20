from typing import Literal

import cupy as cp
import numpy as np


def cdist(x: np.ndarray, y: np.ndarray, metric: Literal["euclidean", "cosine"] = "euclidean") -> cp.ndarray:
    """CuPy implementation of pairwise distance computation.

    Parameters
    ----------
    x
        Array of shape (n_cells_a, n_features)
    y
        Array of shape (n_cells_b, n_features)
    metric
        The distance metric to use: 'euclidean' (default) or 'cosine'.

    Returns
    -------
    dist
        Array of shape (n_cells_a, n_cells_b)
    """
    if metric not in ["euclidean", "cosine"]:
        raise ValueError("Invalid metric choice, must be one of ['euclidean' or 'cosine'].")

    x = cp.asarray(x, dtype=cp.float32)
    y = cp.asarray(y, dtype=cp.float32)

    if metric == "euclidean":
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y^T
        x_sq = cp.sum(x**2, axis=1, keepdims=True)
        y_sq = cp.sum(y**2, axis=1, keepdims=True)
        dist_sq = x_sq + y_sq.T - 2.0 * x @ y.T
        # Clamp to avoid negative values from floating point errors
        dist_sq = cp.maximum(dist_sq, 0.0)
        return cp.sqrt(dist_sq)
    else:
        # cosine distance = 1 - (x @ y^T) / (||x|| * ||y||)
        x_norm = cp.linalg.norm(x, axis=1, keepdims=True)
        y_norm = cp.linalg.norm(y, axis=1, keepdims=True)
        similarity = (x @ y.T) / (x_norm * y_norm.T)
        dist = 1.0 - similarity
        return cp.clip(dist, 0.0, 2.0)


def pdist_squareform(X: np.ndarray) -> cp.ndarray:
    """CuPy implementation of pairwise euclidean distance matrix.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features)

    Returns
    -------
    dist
        Array of shape (n_cells, n_cells)
    """
    X = cp.asarray(X, dtype=cp.float32)
    x_sq = cp.sum(X**2, axis=1, keepdims=True)
    dist_sq = x_sq + x_sq.T - 2.0 * X @ X.T
    dist_sq = cp.maximum(dist_sq, 0.0)
    dist = cp.sqrt(dist_sq)
    # Ensure symmetry and zero diagonal
    dist = (dist + dist.T) / 2.0
    cp.fill_diagonal(dist, 0.0)
    return dist
