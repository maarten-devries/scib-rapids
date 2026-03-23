from typing import Literal

import cupy as cp
import numpy as np


def cdist_sq(x: np.ndarray, y: np.ndarray) -> cp.ndarray:
    """Squared euclidean pairwise distances using expansion formula.

    Fast but less numerically stable — use only where exact distances
    are not required (e.g., kmeans cluster assignment).

    Parameters
    ----------
    x
        Array of shape (n_a, d)
    y
        Array of shape (n_b, d)

    Returns
    -------
    dist_sq
        Array of shape (n_a, n_b)
    """
    x = cp.asarray(x, dtype=cp.float32)
    y = cp.asarray(y, dtype=cp.float32)
    x_sq = cp.sum(x ** 2, axis=1, keepdims=True)
    y_sq = cp.sum(y ** 2, axis=1, keepdims=True)
    dist_sq = x_sq + y_sq.T - 2.0 * x @ y.T
    return cp.maximum(dist_sq, 0.0)


def _euclidean_cdist_direct(x: cp.ndarray, y: cp.ndarray, sub_chunk: int = 2048) -> cp.ndarray:
    """Euclidean pairwise distances using the direct formula sqrt(sum((x-y)^2)).

    Sub-chunks over y to limit memory of the (n_a, sub_chunk, d) intermediate.
    """
    n_a = x.shape[0]
    n_b = y.shape[0]
    result = cp.empty((n_a, n_b), dtype=x.dtype)
    for j in range(0, n_b, sub_chunk):
        j_end = min(j + sub_chunk, n_b)
        diff = x[:, None, :] - y[None, j:j_end, :]  # (n_a, sub_chunk, d)
        result[:, j:j_end] = cp.sqrt(cp.sum(diff * diff, axis=2))
    return result


def cdist(x: np.ndarray, y: np.ndarray, metric: Literal["euclidean", "cosine"] = "euclidean") -> cp.ndarray:
    """CuPy implementation of pairwise distance computation.

    Uses the direct formula for numerical stability (matches JAX/scib-metrics).

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
        return _euclidean_cdist_direct(x, y)
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
    return _euclidean_cdist_direct(X, X)
