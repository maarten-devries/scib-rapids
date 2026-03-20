import logging

import cupy as cp
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2

from scib_rapids.nearest_neighbors import NeighborsResults
from scib_rapids.utils import diffusion_nn, get_ndarray

logger = logging.getLogger(__name__)


def _kbet_core(neigh_batch_ids: np.ndarray, batches: np.ndarray, n_batches: int):
    """Core kBET computation using CuPy."""
    neigh_batch_ids_gpu = cp.asarray(neigh_batch_ids)
    batches_gpu = cp.asarray(batches)

    expected_freq = cp.bincount(batches_gpu, minlength=n_batches).astype(cp.float32)
    expected_freq = expected_freq / cp.sum(expected_freq)
    dof = n_batches - 1

    n_cells, k = neigh_batch_ids_gpu.shape
    # Compute observed counts per cell
    observed_counts = cp.zeros((n_cells, n_batches), dtype=cp.float32)
    for b in range(n_batches):
        observed_counts[:, b] = cp.sum(neigh_batch_ids_gpu == b, axis=1)

    expected_counts = expected_freq * k
    test_statistics = cp.sum((observed_counts - expected_counts) ** 2 / expected_counts, axis=1)

    # Use scipy for chi2 CDF (more reliable than cupy for this)
    test_stats_np = get_ndarray(test_statistics)
    p_values = 1 - chi2.cdf(test_stats_np, dof)

    return test_stats_np, p_values


def kbet(X: NeighborsResults, batches: np.ndarray, alpha: float = 0.05) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute kBET.

    Parameters
    ----------
    X
        A NeighborsResults object.
    batches
        Array of shape (n_cells,) representing batch values.
    alpha
        Significance level for the statistical test.

    Returns
    -------
    acceptance_rate
        Kbet acceptance rate of the sample.
    stat_mean
        Kbet chi-square statistics per cell.
    pvalue_mean
        Kbet p-values per cell.
    """
    if len(batches) != len(X.indices):
        raise ValueError("Length of batches does not match number of cells.")
    knn_idx = X.indices
    batches = np.asarray(pd.Categorical(batches).codes)
    neigh_batch_ids = batches[knn_idx]
    n_batches = len(np.unique(batches))
    test_statistics, p_values = _kbet_core(neigh_batch_ids, batches, n_batches)
    acceptance_rate = float((p_values >= alpha).mean())

    return acceptance_rate, test_statistics, p_values


def kbet_per_label(
    X: NeighborsResults,
    batches: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    diffusion_n_comps: int = 100,
    return_df: bool = False,
) -> float | tuple[float, pd.DataFrame]:
    """Compute kBET score per cell type label.

    Parameters
    ----------
    X
        A NeighborsResults object.
    batches
        Array of shape (n_cells,) representing batch values.
    labels
        Array of shape (n_cells,) representing label values.
    alpha
        Significance level for the statistical test.
    diffusion_n_comps
        Number of diffusion components for diffusion distance approximation.
    return_df
        Return dataframe of results in addition to score.

    Returns
    -------
    kbet_score
        Kbet score over all cells.
    """
    if len(batches) != len(X.indices):
        raise ValueError("Length of batches does not match number of cells.")
    if len(labels) != len(X.indices):
        raise ValueError("Length of labels does not match number of cells.")

    size_max = 2**31 - 1
    batches = np.asarray(pd.Categorical(batches).codes)
    labels = np.asarray(labels)

    conn_graph = X.knn_graph_connectivities

    kbet_scores = {"cluster": [], "kBET": []}
    for clus in np.unique(labels):
        mask = labels == clus
        conn_graph_sub = conn_graph[mask, :][:, mask]
        conn_graph_sub.sort_indices()
        n_obs = conn_graph_sub.shape[0]
        batches_sub = batches[mask]

        if np.logical_or(n_obs < 10, len(np.unique(batches_sub)) == 1):
            logger.info(f"{clus} consists of a single batch or is too small. Skip.")
            score = np.nan
        else:
            quarter_mean = np.floor(np.mean(pd.Series(batches_sub).value_counts()) / 4).astype("int")
            k0 = np.min([70, np.max([10, quarter_mean])])
            if k0 * n_obs >= size_max:
                k0 = np.floor(size_max / n_obs).astype("int")

            n_comp, labs = scipy.sparse.csgraph.connected_components(conn_graph_sub, connection="strong")

            if n_comp == 1:
                try:
                    dc = np.min([diffusion_n_comps, n_obs - 1])
                    nn_graph_sub = diffusion_nn(conn_graph_sub, k=k0, n_comps=dc)
                    score, _, _ = kbet(nn_graph_sub, batches=batches_sub, alpha=alpha)
                except ValueError:
                    logger.info("Diffusion distance failed. Skip.")
                    score = 0
            else:
                comp_size = pd.Series(labs).value_counts()
                comp_size_thresh = 3 * k0
                idx_nonan = np.flatnonzero(np.isin(labs, comp_size[comp_size >= comp_size_thresh].index))

                if len(idx_nonan) / len(labs) >= 0.75:
                    conn_graph_sub_sub = conn_graph_sub[idx_nonan, :][:, idx_nonan]
                    conn_graph_sub_sub.sort_indices()
                    try:
                        dc = np.min([diffusion_n_comps, conn_graph_sub_sub.shape[0] - 1])
                        nn_results_sub_sub = diffusion_nn(conn_graph_sub_sub, k=k0, n_comps=dc)
                        score, _, _ = kbet(nn_results_sub_sub, batches=batches_sub[idx_nonan], alpha=alpha)
                    except ValueError:
                        logger.info("Diffusion distance failed. Skip.")
                        score = 0
                else:
                    score = 0

        kbet_scores["cluster"].append(clus)
        kbet_scores["kBET"].append(score)

    kbet_scores = pd.DataFrame.from_dict(kbet_scores)
    kbet_scores = kbet_scores.reset_index(drop=True)

    final_score = np.nanmean(kbet_scores["kBET"])
    if not return_df:
        return final_score
    else:
        return final_score, kbet_scores
