from typing import Literal

import numpy as np

from ._dataclass import NeighborsResults


def cuml(
    X: np.ndarray,
    n_neighbors: int,
    *,
    metric: Literal["euclidean", "minkowski"] = "euclidean",
) -> NeighborsResults:
    """Run exact Euclidean neighbor search on one GPU.

    Parameters
    ----------
    X
        Data matrix of shape (n_cells, n_features), transferred as float32.
    n_neighbors
        Number of neighbors including self. Must be between 1 and n_cells.
    metric
        Use 'euclidean' for the expanded L2 kernel or 'minkowski' for direct L2
        (p=2). The latter avoids cancellation for large-offset coordinates.

    Returns
    -------
    NeighborsResults
        NumPy indices and Euclidean distances, with self first at distance zero.
        Ordering among equally distant neighbors is unspecified.
    """
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional data matrix")
    if not 1 <= n_neighbors <= X.shape[0]:
        raise ValueError("n_neighbors must be between 1 and the number of cells")
    if metric not in {"euclidean", "minkowski"}:
        raise ValueError("metric must be 'euclidean' or 'minkowski' (p=2)")

    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    X_gpu = cp.asarray(X, dtype=cp.float32, order="C")
    estimator = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="brute",
        metric=metric,
        p=2,
        output_type="numpy",
    )
    estimator.fit(X_gpu)
    distances, indices = estimator.kneighbors(X_gpu)

    # Duplicate embeddings can displace self or put it after another zero-distance
    # cell. Retain self plus the closest n_neighbors-1 other cells in either case.
    rows = np.arange(X.shape[0])
    keep = indices != rows[:, None]
    keep[keep.all(axis=1), -1] = False
    ordered_indices = np.empty_like(indices)
    ordered_distances = np.empty_like(distances)
    ordered_indices[:, 0] = rows
    ordered_distances[:, 0] = 0
    ordered_indices[:, 1:] = indices[keep].reshape(len(rows), n_neighbors - 1)
    ordered_distances[:, 1:] = distances[keep].reshape(len(rows), n_neighbors - 1)
    return NeighborsResults(indices=ordered_indices, distances=ordered_distances)
