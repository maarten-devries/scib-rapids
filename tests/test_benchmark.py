"""Benchmark tests comparing runtime between scib-metrics (JAX) and scib-rapids (CuPy/RAPIDS).

These tests generate toy data and time the execution of each metric
from both packages, reporting speedup factors.

Requirements:
    - scib-metrics must be installed (pip install scib-metrics)
    - scib-rapids must be installed (pip install -e .)
    - A CUDA-capable GPU must be available for scib-rapids

Usage:
    pytest tests/test_benchmark.py -v -s
"""

import time
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

# We try to import both packages - tests are skipped if either is unavailable
rapids_available = True
try:
    import scib_rapids
    from scib_rapids.nearest_neighbors import NeighborsResults as RapidsNeighborsResults
except ImportError:
    rapids_available = False

metrics_available = True
try:
    import scib_metrics
    from scib_metrics.nearest_neighbors import NeighborsResults as MetricsNeighborsResults
except ImportError:
    metrics_available = False

both_available = rapids_available and metrics_available
skip_reason = "Both scib-metrics and scib-rapids must be installed"


@dataclass
class BenchmarkResult:
    metric_name: str
    scib_metrics_time: float
    scib_rapids_time: float

    @property
    def speedup(self) -> float:
        if self.scib_rapids_time == 0:
            return float("inf")
        return self.scib_metrics_time / self.scib_rapids_time

    def __repr__(self) -> str:
        return (
            f"{self.metric_name}: "
            f"scib-metrics={self.scib_metrics_time:.4f}s, "
            f"scib-rapids={self.scib_rapids_time:.4f}s, "
            f"speedup={self.speedup:.2f}x"
        )


def _make_test_data(n_cells=500, n_features=50, n_labels=5, n_batches=3, n_neighbors=30):
    """Generate test data for benchmarking."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_cells, n_features)).astype(np.float32)
    labels = rng.integers(0, n_labels, size=(n_cells,))
    batch = rng.integers(0, n_batches, size=(n_cells,))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)

    return X, labels, batch, dists, inds


def _time_fn(fn, *args, warmup=1, repeats=3, **kwargs):
    """Time a function with warmup and multiple repeats."""
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return min(times), result


@pytest.mark.skipif(not both_available, reason=skip_reason)
class TestBenchmarkMetrics:
    """Compare runtime of scib-metrics vs scib-rapids on toy data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.labels, self.batch, self.dists, self.inds = _make_test_data()
        self.results = []

    def _report(self, result: BenchmarkResult):
        self.results.append(result)
        print(f"\n  {result}")

    def test_benchmark_silhouette_label(self):
        t_metrics, _ = _time_fn(scib_metrics.silhouette_label, self.X, self.labels)
        t_rapids, _ = _time_fn(scib_rapids.silhouette_label, self.X, self.labels)
        result = BenchmarkResult("silhouette_label", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_silhouette_batch(self):
        t_metrics, _ = _time_fn(scib_metrics.silhouette_batch, self.X, self.labels, self.batch)
        t_rapids, _ = _time_fn(scib_rapids.silhouette_batch, self.X, self.labels, self.batch)
        result = BenchmarkResult("silhouette_batch", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_lisi_knn(self):
        nn_metrics = MetricsNeighborsResults(indices=self.inds, distances=self.dists)
        nn_rapids = RapidsNeighborsResults(indices=self.inds, distances=self.dists)

        t_metrics, _ = _time_fn(scib_metrics.lisi_knn, nn_metrics, self.labels, perplexity=10)
        t_rapids, _ = _time_fn(scib_rapids.lisi_knn, nn_rapids, self.labels, perplexity=10)
        result = BenchmarkResult("lisi_knn", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_ilisi_knn(self):
        nn_metrics = MetricsNeighborsResults(indices=self.inds, distances=self.dists)
        nn_rapids = RapidsNeighborsResults(indices=self.inds, distances=self.dists)

        t_metrics, _ = _time_fn(scib_metrics.ilisi_knn, nn_metrics, self.batch, perplexity=10)
        t_rapids, _ = _time_fn(scib_rapids.ilisi_knn, nn_rapids, self.batch, perplexity=10)
        result = BenchmarkResult("ilisi_knn", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_kbet(self):
        nn_metrics = MetricsNeighborsResults(indices=self.inds, distances=self.dists)
        nn_rapids = RapidsNeighborsResults(indices=self.inds, distances=self.dists)

        t_metrics, _ = _time_fn(scib_metrics.kbet, nn_metrics, self.batch)
        t_rapids, _ = _time_fn(scib_rapids.kbet, nn_rapids, self.batch)
        result = BenchmarkResult("kbet", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_nmi_ari_kmeans(self):
        t_metrics, _ = _time_fn(scib_metrics.nmi_ari_cluster_labels_kmeans, self.X, self.labels)
        t_rapids, _ = _time_fn(scib_rapids.nmi_ari_cluster_labels_kmeans, self.X, self.labels)
        result = BenchmarkResult("nmi_ari_kmeans", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_isolated_labels(self):
        t_metrics, _ = _time_fn(scib_metrics.isolated_labels, self.X, self.labels, self.batch)
        t_rapids, _ = _time_fn(scib_rapids.isolated_labels, self.X, self.labels, self.batch)
        result = BenchmarkResult("isolated_labels", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_graph_connectivity(self):
        nn_metrics = MetricsNeighborsResults(indices=self.inds, distances=self.dists)
        nn_rapids = RapidsNeighborsResults(indices=self.inds, distances=self.dists)

        t_metrics, _ = _time_fn(scib_metrics.graph_connectivity, nn_metrics, self.labels)
        t_rapids, _ = _time_fn(scib_rapids.graph_connectivity, nn_rapids, self.labels)
        result = BenchmarkResult("graph_connectivity", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_pcr_comparison(self):
        rng = np.random.default_rng(42)
        X_post = rng.normal(size=self.X.shape).astype(np.float32)
        covariate = self.batch

        t_metrics, _ = _time_fn(scib_metrics.pcr_comparison, self.X, X_post, covariate, categorical=True)
        t_rapids, _ = _time_fn(scib_rapids.pcr_comparison, self.X, X_post, covariate, categorical=True)
        result = BenchmarkResult("pcr_comparison", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_cdist(self):
        t_metrics, _ = _time_fn(scib_metrics.utils.cdist, self.X, self.X)
        t_rapids, _ = _time_fn(scib_rapids.utils.cdist, self.X, self.X)
        result = BenchmarkResult("cdist", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_pca(self):
        t_metrics, _ = _time_fn(scib_metrics.utils.pca, self.X, n_components=10)
        t_rapids, _ = _time_fn(scib_rapids.utils.pca, self.X, n_components=10)
        result = BenchmarkResult("pca", t_metrics, t_rapids)
        self._report(result)

    def test_benchmark_kmeans(self):
        def _run_metrics():
            km = scib_metrics.utils.KMeans(n_clusters=5)
            km.fit(self.X)
            return km

        def _run_rapids():
            km = scib_rapids.utils.KMeans(n_clusters=5)
            km.fit(self.X)
            return km

        t_metrics, _ = _time_fn(_run_metrics)
        t_rapids, _ = _time_fn(_run_rapids)
        result = BenchmarkResult("kmeans", t_metrics, t_rapids)
        self._report(result)


@pytest.mark.skipif(not both_available, reason=skip_reason)
class TestNumericalAgreement:
    """Verify that scib-rapids and scib-metrics produce numerically close results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.labels, self.batch, self.dists, self.inds = _make_test_data(n_cells=100)

    def test_silhouette_label_agreement(self):
        s_metrics = scib_metrics.silhouette_label(self.X, self.labels)
        s_rapids = scib_rapids.silhouette_label(self.X, self.labels)
        np.testing.assert_allclose(s_metrics, s_rapids, atol=1e-3)

    def test_silhouette_batch_agreement(self):
        s_metrics = scib_metrics.silhouette_batch(self.X, self.labels, self.batch)
        s_rapids = scib_rapids.silhouette_batch(self.X, self.labels, self.batch)
        np.testing.assert_allclose(s_metrics, s_rapids, atol=1e-3)

    def test_kbet_agreement(self):
        nn_metrics = MetricsNeighborsResults(indices=self.inds, distances=self.dists)
        nn_rapids = RapidsNeighborsResults(indices=self.inds, distances=self.dists)

        acc_metrics, _, _ = scib_metrics.kbet(nn_metrics, self.batch)
        acc_rapids, _, _ = scib_rapids.kbet(nn_rapids, self.batch)
        np.testing.assert_allclose(acc_metrics, acc_rapids, atol=1e-2)

    def test_lisi_agreement(self):
        nn_metrics = MetricsNeighborsResults(indices=self.inds, distances=self.dists)
        nn_rapids = RapidsNeighborsResults(indices=self.inds, distances=self.dists)

        lisi_metrics = scib_metrics.lisi_knn(nn_metrics, self.labels, perplexity=10)
        lisi_rapids = scib_rapids.lisi_knn(nn_rapids, self.labels, perplexity=10)
        np.testing.assert_allclose(lisi_metrics, lisi_rapids, rtol=1e-3, atol=1e-3)

    def test_graph_connectivity_agreement(self):
        nn_metrics = MetricsNeighborsResults(indices=self.inds, distances=self.dists)
        nn_rapids = RapidsNeighborsResults(indices=self.inds, distances=self.dists)

        gc_metrics = scib_metrics.graph_connectivity(nn_metrics, self.labels)
        gc_rapids = scib_rapids.graph_connectivity(nn_rapids, self.labels)
        np.testing.assert_allclose(gc_metrics, gc_rapids, atol=1e-5)

    def test_cdist_agreement(self):
        d_metrics = scib_metrics.utils.get_ndarray(scib_metrics.utils.cdist(self.X, self.X))
        d_rapids = scib_rapids.utils.get_ndarray(scib_rapids.utils.cdist(self.X, self.X))
        np.testing.assert_allclose(d_metrics, d_rapids, atol=1e-4)

    def test_pca_agreement(self):
        pca_metrics = scib_metrics.utils.pca(self.X, n_components=5)
        pca_rapids = scib_rapids.utils.pca(self.X, n_components=5)
        np.testing.assert_allclose(
            pca_metrics.variance_ratio, pca_rapids.variance_ratio, atol=1e-4
        )
