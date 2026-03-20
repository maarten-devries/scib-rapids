import cupy as cp
import numpy as np

from ._utils import get_ndarray

NdArray = np.ndarray | cp.ndarray


def _Hbeta_batch(
    knn_dists: cp.ndarray,
    self_mask: cp.ndarray,
    beta: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute H and P for all cells simultaneously.

    Parameters
    ----------
    knn_dists
        (n_cells, n_neighbors)
    self_mask
        (n_cells, n_neighbors) boolean — True for non-self neighbors
    beta
        (n_cells,) current beta values

    Returns
    -------
    H
        (n_cells,) entropy values
    P
        (n_cells, n_neighbors) probability matrix
    """
    # beta[:, None] broadcasts to (n_cells, n_neighbors)
    P = cp.exp(-knn_dists * beta[:, None])
    P = cp.where(self_mask, P, 0.0)
    sumP = cp.sum(P, axis=1)  # (n_cells,)

    # Branchless: compute both paths, select via where
    safe_sumP = cp.where(sumP == 0, 1.0, sumP)  # avoid div-by-zero
    H = cp.log(safe_sumP) + beta * cp.sum(knn_dists * P, axis=1) / safe_sumP
    P_normed = P / safe_sumP[:, None]

    # Zero out where sumP == 0
    zero_mask = sumP == 0  # (n_cells,)
    H = cp.where(zero_mask, 0.0, H)
    P_normed = cp.where(zero_mask[:, None], 0.0, P_normed)

    return H, P_normed


def _get_neighbor_probabilities(
    knn_dists: cp.ndarray,
    self_mask: cp.ndarray,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Binary search for perplexity across all cells simultaneously.

    Parameters
    ----------
    knn_dists
        (n_cells, n_neighbors)
    self_mask
        (n_cells, n_neighbors)
    perplexity
        Target perplexity
    tol
        Convergence tolerance for Hdiff
    max_iter
        Maximum binary search iterations

    Returns
    -------
    H
        (n_cells,) final entropy
    P
        (n_cells, n_neighbors) final probabilities
    """
    n_cells = knn_dists.shape[0]
    logU = cp.float32(np.log(perplexity))

    beta = cp.ones(n_cells, dtype=cp.float32)
    betamin = cp.full(n_cells, -cp.inf, dtype=cp.float32)
    betamax = cp.full(n_cells, cp.inf, dtype=cp.float32)

    H, P = _Hbeta_batch(knn_dists, self_mask, beta)
    Hdiff = H - logU

    for _ in range(max_iter):
        # Check which cells still need updating
        active = cp.abs(Hdiff) >= tol  # (n_cells,)
        if not cp.any(active):
            break

        pos = Hdiff > 0  # H too high → increase beta
        neg = ~pos

        # Update betamin/betamax (branchless, like JAX's jnp.where)
        betamin = cp.where(active & pos, beta, betamin)
        betamax = cp.where(active & neg, beta, betamax)

        # Update beta
        new_beta_pos = cp.where(betamax == cp.inf, beta * 2, (beta + betamax) / 2)
        new_beta_neg = cp.where(betamin == -cp.inf, beta / 2, (beta + betamin) / 2)
        new_beta = cp.where(pos, new_beta_pos, new_beta_neg)
        beta = cp.where(active, new_beta, beta)

        H, P = _Hbeta_batch(knn_dists, self_mask, beta)
        Hdiff = H - logU

    return H, P


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

    Fully vectorized across all cells — no Python loops.

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

    n_cells, n_neighbors = knn_dists.shape
    knn_labels = labels[knn_idx]  # (n_cells, n_neighbors)
    self_mask = knn_idx != row_idx  # (n_cells, n_neighbors)

    # Vectorized binary search for all cells
    H, P = _get_neighbor_probabilities(knn_dists, self_mask, perplexity, tol)

    # Compute Simpson index: sum of squared per-label probabilities
    # Build one-hot labels: (n_cells, n_neighbors, n_labels)
    label_onehot = cp.eye(n_labels, dtype=cp.float32)[knn_labels]

    # Weighted sum per label: (n_cells, n_labels)
    sumP_per_label = cp.einsum("ij,ijk->ik", P, label_onehot)

    # Simpson = sum of squares of per-label probabilities
    simpson = cp.sum(sumP_per_label ** 2, axis=1)  # (n_cells,)

    # Where H == 0, set simpson to -1
    simpson = cp.where(H == 0, -1.0, simpson)

    return get_ndarray(simpson)
