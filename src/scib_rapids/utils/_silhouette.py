from typing import Literal

import cupy as cp
import numpy as np
import pandas as pd

from ._utils import get_ndarray


def silhouette_samples(
    X: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> np.ndarray:
    """Compute the Silhouette Coefficient for each observation using cuML.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values.
    chunk_size
        Ignored (kept for API compatibility). cuML handles chunking internally.
    metric
        The distance metric: 'euclidean' (default) or 'cosine'.
    between_cluster_distances
        Only 'nearest' is supported by cuML. Passing other values falls back
        to a chunked CuPy implementation.

    Returns
    -------
    silhouette scores array of shape (n_cells,)
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels should have the same number of samples")

    if between_cluster_distances != "nearest":
        return _silhouette_samples_chunked(X, labels, chunk_size, metric, between_cluster_distances)

    labels_codes = np.asarray(pd.Categorical(labels).codes, dtype=np.int32)

    from cuml.metrics.cluster import silhouette_samples as cuml_silhouette_samples

    X_gpu = cp.asarray(X, dtype=cp.float32)
    labels_gpu = cp.asarray(labels_codes)

    scores = cuml_silhouette_samples(X_gpu, labels_gpu, metric=metric)
    return get_ndarray(scores)


def _silhouette_samples_chunked(
    X: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> np.ndarray:
    """Fallback for non-'nearest' between_cluster_distances using chunked pairwise distances."""
    from functools import partial

    from ._dist import cdist

    labels_codes = pd.Categorical(labels).codes
    labels_gpu = cp.asarray(labels_codes)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    label_freqs = cp.bincount(labels_gpu)

    reduce_fn = partial(
        _silhouette_reduce,
        labels=labels_gpu,
        label_freqs=label_freqs,
        between_cluster_distances=between_cluster_distances,
    )

    n_samples = X_gpu.shape[0]
    n_chunks = int(np.ceil(n_samples / chunk_size))
    intra_dists_all = []
    inter_dists_all = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_samples)
        D_chunk = cdist(X_gpu[start:end], X_gpu, metric=metric)
        intra, inter = reduce_fn(D_chunk, start=start)
        intra_dists_all.append(intra)
        inter_dists_all.append(inter)

    intra_clust_dists = cp.concatenate(intra_dists_all)
    inter_clust_dists = cp.concatenate(inter_dists_all)

    denom = (label_freqs - 1)[labels_gpu]
    intra_clust_dists = intra_clust_dists / denom
    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples = sil_samples / cp.maximum(intra_clust_dists, inter_clust_dists)
    return get_ndarray(cp.nan_to_num(sil_samples))


def _silhouette_reduce(
    D_chunk: cp.ndarray,
    start: int,
    labels: cp.ndarray,
    label_freqs: cp.ndarray,
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> tuple[cp.ndarray, cp.ndarray]:
    """Accumulate silhouette statistics for vertical chunk of X."""
    D_chunk_len = D_chunk.shape[0]
    n_labels = label_freqs.shape[0]

    clust_dists = cp.zeros((D_chunk_len, n_labels), dtype=D_chunk.dtype)
    for i in range(n_labels):
        mask = labels == i
        clust_dists[:, i] = cp.sum(D_chunk[:, mask], axis=1)

    chunk_labels = labels[start : start + D_chunk_len]
    row_idx = cp.arange(D_chunk_len)
    intra_clust_dists = clust_dists[row_idx, chunk_labels]

    if between_cluster_distances == "furthest":
        clust_dists[row_idx, chunk_labels] = -cp.inf
        clust_dists = clust_dists / label_freqs
        inter_clust_dists = cp.max(clust_dists, axis=1)
    elif between_cluster_distances == "mean_other":
        clust_dists[row_idx, chunk_labels] = cp.nan
        total_other_dists = cp.nansum(clust_dists, axis=1)
        total_other_count = cp.sum(label_freqs) - label_freqs[chunk_labels]
        inter_clust_dists = total_other_dists / total_other_count
    elif between_cluster_distances == "nearest":
        clust_dists[row_idx, chunk_labels] = cp.inf
        clust_dists = clust_dists / label_freqs
        inter_clust_dists = cp.min(clust_dists, axis=1)
    else:
        raise ValueError("Parameter 'between_cluster_distances' must be one of ['nearest', 'mean_other', 'furthest'].")
    return intra_clust_dists, inter_clust_dists
