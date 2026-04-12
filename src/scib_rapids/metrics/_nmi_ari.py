import logging
import warnings

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import check_array

from scib_rapids.nearest_neighbors import NeighborsResults
from scib_rapids.utils import KMeans

logger = logging.getLogger(__name__)


def _compute_clustering_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    return kmeans.labels_


def _compute_clustering_leiden(connectivity_graph: spmatrix, resolution: float, seed: int) -> np.ndarray:
    import cudf
    import cugraph

    coo = cpsp.coo_matrix(connectivity_graph)
    n = coo.shape[0]
    sources = cp.asarray(coo.row)
    targets = cp.asarray(coo.col)
    weights = cp.asarray(coo.data, dtype=cp.float32)

    edges = cudf.DataFrame({"src": sources, "dst": targets, "weight": weights})
    g = cugraph.Graph(directed=False)
    g.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="weight", renumber=False)

    parts, _ = cugraph.leiden(g, resolution=resolution, random_state=seed)
    parts = parts.sort_values("vertex")
    clusters = cp.zeros(n, dtype=np.int64)
    clusters[parts["vertex"].to_cupy()] = parts["partition"].to_cupy()
    return cp.asnumpy(clusters)


def _compute_nmi_ari_cluster_labels(
    X: spmatrix,
    labels: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> tuple[float, float]:
    labels_pred = _compute_clustering_leiden(X, resolution, seed)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)
    return nmi, ari


def nmi_ari_cluster_labels_kmeans(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute NMI and ARI between k-means clusters and labels.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values.

    Returns
    -------
    dict with 'nmi' and 'ari' keys
    """
    X = check_array(X, accept_sparse=False, ensure_2d=True)
    n_clusters = len(np.unique(labels))
    labels_pred = _compute_clustering_kmeans(X, n_clusters)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)

    return {"nmi": nmi, "ari": ari}


def nmi_ari_cluster_labels_leiden(
    X: NeighborsResults,
    labels: np.ndarray,
    optimize_resolution: bool = True,
    resolution: float = 1.0,
    n_jobs: int = 1,
    seed: int = 42,
) -> dict[str, float]:
    """Compute NMI and ARI between leiden clusters and labels.

    Parameters
    ----------
    X
        A NeighborsResults object.
    labels
        Array of shape (n_cells,) representing label values.
    optimize_resolution
        Whether to optimize the resolution parameter.
    resolution
        Resolution parameter (used if optimize_resolution is False).
    n_jobs
        Number of jobs for parallelization.
    seed
        Seed for reproducibility.

    Returns
    -------
    dict with 'nmi' and 'ari' keys
    """
    conn_graph = X.knn_graph_connectivities
    if optimize_resolution:
        n = 10
        resolutions = np.array([2 * x / n for x in range(1, n + 1)])
        try:
            from joblib import Parallel, delayed

            out = Parallel(n_jobs=n_jobs)(
                delayed(_compute_nmi_ari_cluster_labels)(conn_graph, labels, r, seed=seed) for r in resolutions
            )
        except ImportError:
            warnings.warn("Using for loop over clustering resolutions. `pip install joblib` for parallelization.")
            out = [_compute_nmi_ari_cluster_labels(conn_graph, labels, r, seed=seed) for r in resolutions]
        nmi_ari = np.array(out)
        nmi_ind = np.argmax(nmi_ari[:, 0])
        nmi, ari = nmi_ari[nmi_ind, :]
        return {"nmi": nmi, "ari": ari}
    else:
        nmi, ari = _compute_nmi_ari_cluster_labels(conn_graph, labels, resolution, seed=seed)

    return {"nmi": nmi, "ari": ari}
