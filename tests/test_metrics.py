"""Unit tests for scib-rapids metrics.

Tests verify that scib-rapids produces correct results by checking
output types, shapes, and value ranges on toy data.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples as sk_silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors

import scib_rapids
from scib_rapids.nearest_neighbors import NeighborsResults
from tests.utils.data import dummy_x_labels, dummy_x_labels_batch


def test_cdist():
    from scipy.spatial.distance import cdist as sp_cdist

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    result = scib_rapids.utils.get_ndarray(scib_rapids.utils.cdist(x, y))
    assert np.allclose(result, sp_cdist(x, y), atol=1e-5)


def test_cdist_cosine():
    from scipy.spatial.distance import cdist as sp_cdist

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    result = scib_rapids.utils.get_ndarray(scib_rapids.utils.cdist(x, y, metric="cosine"))
    assert np.allclose(result, sp_cdist(x, y, metric="cosine"), atol=1e-5)


def test_pdist():
    from scipy.spatial.distance import pdist, squareform

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    result = scib_rapids.utils.get_ndarray(scib_rapids.utils.pdist_squareform(x))
    assert np.allclose(result, squareform(pdist(x)), atol=1e-5)


def test_silhouette_samples():
    X, labels = dummy_x_labels()
    rapids_sil = scib_rapids.utils.silhouette_samples(X, labels)
    sk_sil = sk_silhouette_samples(X, labels)
    assert np.allclose(rapids_sil, sk_sil, atol=1e-4)


def test_silhouette_label():
    X, labels = dummy_x_labels()
    score = scib_rapids.silhouette_label(X, labels)
    assert 0 <= score <= 1
    score_unscaled = scib_rapids.silhouette_label(X, labels, rescale=False)
    assert -1 <= score_unscaled <= 1


def test_silhouette_batch():
    X, labels, batch = dummy_x_labels_batch()
    score = scib_rapids.silhouette_batch(X, labels, batch)
    assert score > 0


def test_lisi_knn():
    X, labels = dummy_x_labels()
    nbrs = NearestNeighbors(n_neighbors=30, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)
    neigh_results = NeighborsResults(indices=inds, distances=dists)
    lisi_res = scib_rapids.lisi_knn(neigh_results, labels, perplexity=10)
    assert lisi_res.shape == (100,)
    assert np.all(np.isfinite(lisi_res))


def test_ilisi_clisi_knn():
    X, labels, batches = dummy_x_labels_batch(x_is_neighbors_results=True)
    ilisi = scib_rapids.ilisi_knn(X, batches, perplexity=10)
    clisi = scib_rapids.clisi_knn(X, labels, perplexity=10)
    assert isinstance(ilisi, (float, np.floating))
    assert isinstance(clisi, (float, np.floating))


def test_nmi_ari_cluster_labels_kmeans():
    X, labels = dummy_x_labels()
    out = scib_rapids.nmi_ari_cluster_labels_kmeans(X, labels)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_nmi_ari_cluster_labels_leiden():
    X, labels = dummy_x_labels(symmetric_positive=True, x_is_neighbors_results=True)
    out = scib_rapids.nmi_ari_cluster_labels_leiden(X, labels, optimize_resolution=False, resolution=0.1)
    nmi, ari = out["nmi"], out["ari"]
    assert isinstance(nmi, float)
    assert isinstance(ari, float)


def test_kbet():
    X, _, batch = dummy_x_labels_batch(x_is_neighbors_results=True)
    acc_rate, stats, pvalues = scib_rapids.kbet(X, batch)
    assert isinstance(acc_rate, float)
    assert len(stats) == X.indices.shape[0]
    assert len(pvalues) == X.indices.shape[0]


def test_kbet_per_label():
    X, labels, batch = dummy_x_labels_batch(x_is_neighbors_results=True)
    score = scib_rapids.kbet_per_label(X, batch, labels)
    assert isinstance(score, (float, np.floating))


def test_graph_connectivity():
    X, labels = dummy_x_labels(symmetric_positive=True, x_is_neighbors_results=True)
    metric = scib_rapids.graph_connectivity(X, labels)
    assert isinstance(metric, (float, np.floating))


def test_isolated_labels():
    X, labels, batch = dummy_x_labels_batch()
    score = scib_rapids.isolated_labels(X, labels, batch)
    assert isinstance(score, (float, np.floating))


def test_kmeans():
    centers = np.array([[1, 1], [-1, -1], [1, -1]], dtype=np.float32) * 2
    X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, random_state=42)
    X = X.astype(np.float32)

    kmeans = scib_rapids.utils.KMeans(n_clusters=3)
    kmeans.fit(X)
    assert kmeans.labels_.shape == (X.shape[0],)

    from sklearn.cluster import KMeans as SKMeans

    skmeans = SKMeans(n_clusters=3)
    skmeans.fit(X)

    # Reorder cluster centroids and measure accuracy
    order = pairwise_distances_argmin(kmeans.cluster_centroids_, skmeans.cluster_centers_)
    sk_means_cluster_centers = skmeans.cluster_centers_[order]

    k_means_labels = pairwise_distances_argmin(X, kmeans.cluster_centroids_)
    sk_means_labels = pairwise_distances_argmin(X, sk_means_cluster_centers)

    accuracy = (k_means_labels == sk_means_labels).sum() / len(k_means_labels)
    assert accuracy > 0.99


def test_pca():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 10)).astype(np.float32)

    rapids_pca = scib_rapids.utils.pca(X, n_components=5)
    sk_pca = PCA(n_components=5).fit(X)

    assert rapids_pca.coordinates.shape == (50, 5)
    assert rapids_pca.components.shape == (5, 10)
    # Variance ratios should be close
    np.testing.assert_allclose(rapids_pca.variance_ratio, sk_pca.explained_variance_ratio_, atol=1e-4)


def test_pcr_comparison():
    rng = np.random.default_rng(42)
    X_pre = rng.normal(size=(100, 10)).astype(np.float32)
    X_post = rng.normal(size=(100, 10)).astype(np.float32)
    covariate = rng.integers(0, 3, size=(100,))
    score = scib_rapids.pcr_comparison(X_pre, X_post, covariate, categorical=True)
    assert isinstance(score, (float, int))
