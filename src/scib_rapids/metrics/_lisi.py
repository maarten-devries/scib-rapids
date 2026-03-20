import numpy as np
import pandas as pd

from scib_rapids.nearest_neighbors import NeighborsResults
from scib_rapids.utils import compute_simpson_index


def lisi_knn(X: NeighborsResults, labels: np.ndarray, perplexity: float = None) -> np.ndarray:
    """Compute the local inverse Simpson index (LISI) for each cell.

    Parameters
    ----------
    X
        A NeighborsResults object.
    labels
        Array of shape (n_cells,) representing label values.
    perplexity
        Parameter controlling effective neighborhood size.

    Returns
    -------
    lisi
        Array of shape (n_cells,).
    """
    labels = np.asarray(pd.Categorical(labels).codes)
    knn_dists, knn_idx = X.distances, X.indices
    row_idx = np.arange(X.n_samples)[:, np.newaxis]

    if perplexity is None:
        perplexity = np.floor(knn_idx.shape[1] / 3)

    n_labels = len(np.unique(labels))

    simpson = compute_simpson_index(
        knn_dists=knn_dists, knn_idx=knn_idx, row_idx=row_idx, labels=labels, n_labels=n_labels, perplexity=perplexity
    )
    return 1 / simpson


def ilisi_knn(X: NeighborsResults, batches: np.ndarray, perplexity: float = None, scale: bool = True) -> float:
    """Compute the integration LISI (iLISI).

    Parameters
    ----------
    X
        A NeighborsResults object.
    batches
        Array of shape (n_cells,) representing batch values.
    perplexity
        Parameter controlling effective neighborhood size.
    scale
        Scale lisi into the range [0, 1].

    Returns
    -------
    ilisi score
    """
    batches = np.asarray(pd.Categorical(batches).codes)
    lisi = lisi_knn(X, batches, perplexity=perplexity)
    ilisi = np.nanmedian(lisi)
    if scale:
        nbatches = len(np.unique(batches))
        ilisi = (ilisi - 1) / (nbatches - 1)
    return ilisi


def clisi_knn(X: NeighborsResults, labels: np.ndarray, perplexity: float = None, scale: bool = True) -> float:
    """Compute the cell-type LISI (cLISI).

    Parameters
    ----------
    X
        A NeighborsResults object.
    labels
        Array of shape (n_cells,) representing cell type label values.
    perplexity
        Parameter controlling effective neighborhood size.
    scale
        Scale lisi into the range [0, 1].

    Returns
    -------
    clisi score
    """
    labels = np.asarray(pd.Categorical(labels).codes)
    lisi = lisi_knn(X, labels, perplexity=perplexity)
    clisi = np.nanmedian(lisi)
    if scale:
        nlabels = len(np.unique(labels))
        clisi = (nlabels - clisi) / (nlabels - 1)
    return clisi
