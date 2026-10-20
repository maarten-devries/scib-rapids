"""Exact GPU neighbor references and diffusion kBET dispatch tests."""

import importlib
from types import SimpleNamespace

import cupy as cp
import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix
from scipy.spatial.distance import cdist

import scib_rapids as scib

diffusion_module = importlib.import_module("scib_rapids.utils._diffusion_nn")
kbet_module = importlib.import_module("scib_rapids.metrics._kbet")


@pytest.mark.parametrize("metric", ["euclidean", "minkowski"])
@pytest.mark.parametrize(
    "cells,count,duplicates",
    [(1, 1, False), (2, 2, False), (120, 1, False), (120, 15, False), (120, 15, True), (120, 120, True)],
)
def test_cuml_neighbors_match_exact_cpu_distances(metric, cells, count, duplicates):
    """Honor k including self, duplicate ties, and Euclidean distance units."""
    X = np.random.default_rng(42).normal(size=(cells, 7)).astype(np.float32)
    if duplicates:
        X[:20] = X[0]
    actual = scib.nearest_neighbors.cuml(X, n_neighbors=count, metric=metric)
    reference = cdist(X, X)
    expected_distances = np.sort(reference, axis=1)[:, :count]
    selected_distances = np.take_along_axis(reference, actual.indices, axis=1)
    assert actual.indices.shape == actual.distances.shape == (cells, count)
    assert actual.distances.dtype == np.float32
    np.testing.assert_array_equal(actual.indices[:, 0], np.arange(cells))
    np.testing.assert_array_equal(actual.distances[:, 0], np.zeros(cells))
    np.testing.assert_allclose(actual.distances, expected_distances, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual.distances, selected_distances, atol=2e-5, rtol=2e-5)
    assert all(len(set(row)) == count for row in actual.indices)


def test_direct_l2_neighbors_preserve_large_offset_precision():
    """Diffusion distances must not collapse through expanded L2 cancellation."""
    X = (np.random.default_rng(42).normal(size=(256, 32)) + 10000).astype(np.float32)
    actual = scib.nearest_neighbors.cuml(X, n_neighbors=20, metric="minkowski")
    reference = cdist(X, X)
    np.testing.assert_allclose(actual.distances, np.sort(reference, axis=1)[:, :20], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual.distances, np.take_along_axis(reference, actual.indices, axis=1), atol=2e-5)


@pytest.mark.parametrize("count", [0, -1, 5])
def test_cuml_neighbors_validate_counts(count):
    """Do not silently truncate the requested library neighbor count."""
    with pytest.raises(ValueError, match="n_neighbors"):
        scib.nearest_neighbors.cuml(np.ones((4, 2)), n_neighbors=count)


def test_cuml_neighbors_transfer_contiguous_float32(monkeypatch):
    """Transfer once, then query that same matrix with explicit output types."""
    calls = []

    class Estimator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit(self, X):
            assert X.dtype == cp.float32
            assert X.flags.c_contiguous
            self.X = X

        def kneighbors(self, X):
            assert X is self.X
            # Coincident points may displace self from a truncated neighbor list.
            return np.zeros((4, 3), dtype=np.float32), np.tile(np.arange(3), (4, 1))

    monkeypatch.setattr("cuml.neighbors.NearestNeighbors", Estimator)
    result = scib.nearest_neighbors.cuml(np.ones((4, 6), dtype=np.float64)[:, ::2], n_neighbors=3)
    assert calls == [{"n_neighbors": 3, "algorithm": "brute", "metric": "euclidean", "p": 2, "output_type": "numpy"}]
    np.testing.assert_array_equal(result.indices[:, 0], np.arange(4))
    assert all(len(set(row)) == 3 for row in result.indices)


@pytest.mark.parametrize("backend", ["pynndescent", "cuml"])
def test_diffusion_dispatch_keeps_components_scaling_and_neighbor_count(monkeypatch, backend):
    """Change search only, preserving the eigensolver and k+1 convention."""
    eigenvectors = np.arange(12, dtype=np.float32).reshape(4, 3)
    eigenvalues = np.array([1, 0.5, 0.25], dtype=np.float32)
    calls = []
    monkeypatch.setattr(diffusion_module, "_compute_transitions", lambda graph: graph)
    monkeypatch.setattr(
        diffusion_module, "_compute_eigen", lambda graph, n_comps: (eigenvalues.copy(), eigenvectors.copy())
    )

    def neighbors(X, n_neighbors, **kwargs):
        calls.append((X, n_neighbors, kwargs))
        return "neighbors"

    def unexpected(*args, **kwargs):
        pytest.fail("The unselected neighbor backend must not run")

    monkeypatch.setattr(scib.nearest_neighbors, backend, neighbors)
    monkeypatch.setattr(scib.nearest_neighbors, "cuml" if backend == "pynndescent" else "pynndescent", unexpected)
    assert scib.utils.diffusion_nn(csr_matrix(np.eye(4)), k=2, n_comps=3, neighbor_backend=backend) == "neighbors"
    expected_values = eigenvalues + 1e-8
    expected = eigenvectors * np.array([v if v == 1 else v / (1 - v) for v in expected_values])
    np.testing.assert_allclose(calls[0][0], expected)
    assert calls[0][1:] == (3, {"metric": "minkowski"} if backend == "cuml" else {})


@pytest.mark.parametrize("backend", ["pynndescent", "cuml"])
@pytest.mark.parametrize("disconnected", [False, True])
def test_kbet_forwards_backend_in_both_graph_branches(monkeypatch, backend, disconnected):
    """Forward selection through retained connected components and normal graphs."""
    graph = (
        block_diag([np.ones((40, 40)), np.ones((40, 40))], format="csr")
        if disconnected
        else csr_matrix(np.ones((80, 80)))
    )
    neighbors = SimpleNamespace(indices=np.zeros((80, 11), dtype=int), knn_graph_connectivities=graph)
    calls = []

    def diffusion(graph, **kwargs):
        calls.append((graph.shape, kwargs))
        return neighbors

    monkeypatch.setattr(kbet_module, "diffusion_nn", diffusion)
    monkeypatch.setattr(kbet_module, "kbet", lambda *args, **kwargs: (0.75, None, None))
    score, frame = scib.kbet_per_label(
        neighbors, np.arange(80) % 2, np.zeros(80), return_df=True, neighbor_backend=backend
    )
    assert score == 0.75
    assert frame["kBET"].tolist() == [0.75]
    assert calls == [((80, 80), {"k": 10, "n_comps": 79, "neighbor_backend": backend})]


def test_kbet_default_and_skip_rules(monkeypatch):
    """Keep the CPU default and avoid diffusion for undefined labels."""
    graph = block_diag([np.ones((24, 24)), np.ones((4, 4)), np.ones((12, 12))], format="csr")
    neighbors = SimpleNamespace(indices=np.zeros((40, 11), dtype=int), knn_graph_connectivities=graph)
    labels = np.repeat([0, 1, 2], [24, 4, 12])
    batches = np.r_[np.arange(28) % 2, np.zeros(12)]
    calls = []

    def diffusion(graph, **kwargs):
        calls.append(kwargs)
        return neighbors

    monkeypatch.setattr(kbet_module, "diffusion_nn", diffusion)
    monkeypatch.setattr(kbet_module, "kbet", lambda *args, **kwargs: (0.75, None, None))
    score, frame = scib.kbet_per_label(neighbors, batches, labels, return_df=True)
    assert score == 0.75
    assert frame["kBET"].isna().tolist() == [False, True, True]
    assert calls == [{"k": 10, "n_comps": 23, "neighbor_backend": "pynndescent"}]


def test_unknown_backend_fails_before_diffusion_error_handling():
    """An invalid backend must not become a valid-looking zero kBET score."""
    with pytest.raises(ValueError, match="neighbor_backend"):
        scib.kbet_per_label(None, np.array([]), np.array([]), neighbor_backend="invalid")
    with pytest.raises(ValueError, match="neighbor_backend"):
        scib.utils.diffusion_nn(None, k=2, neighbor_backend="invalid")


def test_gpu_diffusion_errors_do_not_fall_back(monkeypatch):
    """A failed GPU search never invokes the CPU neighbor implementation."""
    monkeypatch.setattr(diffusion_module, "_compute_transitions", lambda graph: graph)
    monkeypatch.setattr(diffusion_module, "_compute_eigen", lambda *args, **kwargs: (np.array([0.5]), np.ones((4, 1))))

    def unavailable(*args, **kwargs):
        raise ImportError("cuML unavailable")

    def unexpected(*args, **kwargs):
        pytest.fail("CPU fallback must not run")

    monkeypatch.setattr(scib.nearest_neighbors, "cuml", unavailable)
    monkeypatch.setattr(scib.nearest_neighbors, "pynndescent", unexpected)
    with pytest.raises(ImportError, match="cuML unavailable"):
        scib.utils.diffusion_nn(csr_matrix(np.eye(4)), k=2, neighbor_backend="cuml")


@pytest.mark.parametrize("disconnected", [False, True])
def test_gpu_kbet_matches_exact_cpu_diffusion_neighbors(monkeypatch, disconnected):
    """Compare kBET to exact CPU neighbors, not the optimized implementation itself."""
    rng = np.random.default_rng(43)
    points = rng.normal(size=(120, 7))
    distances = cdist(points, points)
    graph = csr_matrix(np.exp(-distances))
    if disconnected:
        graph = block_diag([graph[:60, :60], graph[60:, 60:]], format="csr")
    neighbors = SimpleNamespace(indices=np.zeros((120, 11), dtype=int), knn_graph_connectivities=graph)
    batches = np.arange(120) % 3
    labels = np.zeros(120)
    actual = scib.kbet_per_label(neighbors, batches, labels, neighbor_backend="cuml")

    def exact_cpu(X, n_neighbors):
        distances = cdist(X, X)
        indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        return scib.nearest_neighbors.NeighborsResults(
            indices=indices, distances=np.take_along_axis(distances, indices, axis=1)
        )

    monkeypatch.setattr(scib.nearest_neighbors, "pynndescent", exact_cpu)
    expected = scib.kbet_per_label(neighbors, batches, labels)
    assert actual == pytest.approx(expected, abs=1e-6)
