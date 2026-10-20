from typing import Literal

import numpy as np

from scib_rapids.utils import silhouette_samples


def silhouette_label(
    X: np.ndarray,
    labels: np.ndarray,
    rescale: bool = True,
    chunk_size: int = 4096,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    *,
    precomputed_silhouettes: np.ndarray | None = None,
) -> float:
    """Average silhouette width (ASW).

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values.
    rescale
        Scale asw into the range [0, 1].
    chunk_size
        Size of chunks to process at a time.
    metric
        'euclidean' (default) or 'cosine'.
    precomputed_silhouettes
        Raw per-cell silhouettes computed with the same rows, labels, and metric.
        When provided, skip distance computation. Rescaling is still applied here.

    Returns
    -------
    silhouette score
    """
    if precomputed_silhouettes is None:
        silhouettes = silhouette_samples(X, labels, chunk_size=chunk_size, metric=metric)
    else:
        silhouettes = np.asarray(precomputed_silhouettes)
        if silhouettes.shape != (len(labels),) or X.shape[0] != len(labels):
            raise ValueError("precomputed_silhouettes must be a vector aligned with X and labels")
    asw = np.mean(silhouettes)
    if rescale:
        asw = (asw + 1) / 2
    return float(np.mean(asw))


def silhouette_batch(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    rescale: bool = True,
    chunk_size: int = 4096,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    between_cluster_distances: Literal["nearest", "mean_other", "furthest"] = "nearest",
) -> float:
    """Average silhouette width (ASW) with respect to batch ids within each label.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values.
    batch
        Array of shape (n_cells,) representing batch values.
    rescale
        Scale asw into the range [0, 1].
    chunk_size
        Size of chunks to process at a time.
    metric
        'euclidean' (default) or 'cosine'.
    between_cluster_distances
        'nearest', 'mean_other', or 'furthest'.

    Returns
    -------
    silhouette score
    """
    group_scores = []
    unique_labels = np.unique(labels)
    for group in unique_labels:
        labels_mask = labels == group
        X_subset = X[labels_mask]
        batch_subset = batch[labels_mask]
        n_batches = len(np.unique(batch_subset))

        if (n_batches == 1) or (n_batches == X_subset.shape[0]):
            continue

        sil_per_group = silhouette_samples(
            X_subset,
            batch_subset,
            chunk_size=chunk_size,
            metric=metric,
            between_cluster_distances=between_cluster_distances,
        )

        sil_per_group = np.abs(sil_per_group)
        if rescale:
            sil_per_group = 1 - sil_per_group

        group_scores.append(np.mean(sil_per_group, dtype=np.float64))

    return float(np.mean(group_scores)) if group_scores else float("nan")


def bras(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "cosine",
    between_cluster_distances: Literal["mean_other", "furthest"] = "mean_other",
) -> float:
    """Batch removal adapted silhouette (BRAS).

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values.
    batch
        Array of shape (n_cells,) representing batch values.
    chunk_size
        Size of chunks to process at a time.
    metric
        'euclidean' or 'cosine' (default).
    between_cluster_distances
        'mean_other' (default) or 'furthest'.

    Returns
    -------
    BRAS score
    """
    return silhouette_batch(X, labels, batch, True, chunk_size, metric, between_cluster_distances)
