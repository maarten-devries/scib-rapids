import logging
import warnings

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


def _create_cugraph_from_sparse(connectivity_graph: spmatrix):
    try:
        import cudf
        import cugraph
    except ImportError as err:
        raise ImportError(
            "Leiden clustering requires RAPIDS cuGraph and cuDF. "
            "Install the CUDA 12 packages with `pip install cugraph-cu12 cudf-cu12`."
        ) from err

    coo_graph = connectivity_graph.tocoo(copy=False)
    df = cudf.DataFrame(
        {
            "source": coo_graph.row.astype(np.int32, copy=False),
            "destination": coo_graph.col.astype(np.int32, copy=False),
            "weight": coo_graph.data.astype(np.float32, copy=False),
        }
    )

    graph = cugraph.Graph()
    graph.from_cudf_edgelist(
        df,
        source="source",
        destination="destination",
        weight="weight",
        vertices=cudf.Series(np.arange(connectivity_graph.shape[0], dtype=np.int32)),
    )
    return graph


def _compute_clustering_leiden(connectivity_graph: spmatrix, resolution: float, seed: int) -> np.ndarray:
    if connectivity_graph.shape[0] != connectivity_graph.shape[1]:
        raise ValueError("connectivity_graph must be square")

    if connectivity_graph.nnz == 0:
        return np.arange(connectivity_graph.shape[0])

    try:
        from cugraph import leiden
    except ImportError as err:
        raise ImportError(
            "Leiden clustering requires RAPIDS cuGraph. Install it with `pip install cugraph-cu12`."
        ) from err

    g = _create_cugraph_from_sparse(connectivity_graph)
    clusters, _ = leiden(g, resolution=resolution, random_state=seed)
    clusters = clusters.to_pandas()
    labels = np.arange(connectivity_graph.shape[0])
    vertices = clusters["vertex"].to_numpy(dtype=np.intp)
    labels[vertices] = clusters["partition"].to_numpy()
    return labels


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
        Deprecated. Ignored by the GPU Leiden backend.
    seed
        Seed for reproducibility.

    Returns
    -------
    dict with 'nmi' and 'ari' keys
    """
    conn_graph = X.knn_graph_connectivities
    if optimize_resolution:
        if n_jobs != 1:
            warnings.warn("`n_jobs` is ignored by the GPU Leiden backend.", stacklevel=2)
        n = 10
        resolutions = np.array([2 * x / n for x in range(1, n + 1)])
        out = [_compute_nmi_ari_cluster_labels(conn_graph, labels, r, seed=seed) for r in resolutions]
        nmi_ari = np.array(out)
        nmi_ind = np.argmax(nmi_ari[:, 0])
        nmi, ari = nmi_ari[nmi_ind, :]
        return {"nmi": nmi, "ari": ari}
    else:
        nmi, ari = _compute_nmi_ari_cluster_labels(conn_graph, labels, resolution, seed=seed)

    return {"nmi": nmi, "ari": ari}
