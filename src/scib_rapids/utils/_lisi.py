import cupy as cp
import numpy as np

from ._utils import get_ndarray

NdArray = np.ndarray | cp.ndarray


def _Hbeta(knn_dists_row: cp.ndarray, row_self_mask: cp.ndarray, beta: float) -> tuple[cp.ndarray, cp.ndarray]:
    P = cp.exp(-knn_dists_row * beta)
    P = cp.where(row_self_mask, P, 0.0)
    sumP = cp.nansum(P)
    if sumP == 0:
        H = cp.float32(0.0)
        P = cp.zeros_like(knn_dists_row)
    else:
        H = cp.log(sumP) + beta * cp.nansum(knn_dists_row * P) / sumP
        P = P / sumP
    return H, P


def _get_neighbor_probability(
    knn_dists_row: cp.ndarray, row_self_mask: cp.ndarray, perplexity: float, tol: float
) -> tuple[cp.ndarray, cp.ndarray]:
    beta = 1.0
    betamin = -np.inf
    betamax = np.inf
    H, P = _Hbeta(knn_dists_row, row_self_mask, beta)
    logU = cp.log(cp.float32(perplexity))
    Hdiff = float((H - logU).item())

    tries = 0
    while abs(Hdiff) >= tol and tries < 50:
        if Hdiff > 0:
            betamin = beta
            beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
        else:
            betamax = beta
            beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
        H, P = _Hbeta(knn_dists_row, row_self_mask, beta)
        Hdiff = float((H - logU).item())
        tries += 1

    return H, P


def _compute_simpson_index_cell(
    knn_dists_row: cp.ndarray,
    knn_labels_row: cp.ndarray,
    row_self_mask: cp.ndarray,
    n_batches: int,
    perplexity: float,
    tol: float,
) -> float:
    H, P = _get_neighbor_probability(knn_dists_row, row_self_mask, perplexity, tol)

    if float(H.item()) == 0:
        return -1.0

    sumP = cp.zeros(n_batches, dtype=cp.float32)
    for b in range(n_batches):
        mask = knn_labels_row == b
        sumP[b] = cp.sum(P[mask])
    return float(cp.dot(sumP, sumP).item())


def compute_simpson_index(
    knn_dists: NdArray,
    knn_idx: NdArray,
    row_idx: NdArray,
    labels: NdArray,
    n_labels: int,
    perplexity: float = 30,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute the Simpson index for each cell using CuPy.

    Parameters
    ----------
    knn_dists
        KNN distances of size (n_cells, n_neighbors).
    knn_idx
        KNN indices of size (n_cells, n_neighbors).
    row_idx
        Idx of each row (n_cells, 1).
    labels
        Cell labels of size (n_cells,).
    n_labels
        Number of labels.
    perplexity
        Measure of the effective number of neighbors.
    tol
        Tolerance for binary search.

    Returns
    -------
    simpson_index
        Simpson index of size (n_cells,).
    """
    knn_dists = cp.asarray(knn_dists, dtype=cp.float32)
    knn_idx = cp.asarray(knn_idx)
    labels = cp.asarray(labels)
    row_idx = cp.asarray(row_idx)

    n_cells = knn_dists.shape[0]
    knn_labels = labels[knn_idx]
    self_mask = knn_idx != row_idx

    # Process each cell
    out = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        out[i] = _compute_simpson_index_cell(
            knn_dists[i], knn_labels[i], self_mask[i], n_batches=n_labels, perplexity=perplexity, tol=tol
        )

    return out
