"""Cross-validation tests: verify scib-rapids produces EXACTLY the same results as scib-metrics.

Strategy:
    1. Generate deterministic test data and save to a temp directory as .npy files.
    2. For each metric, run a small Python script in each venv (scib-metrics / scib-rapids)
       that loads the data, computes the metric, and saves the result as .npy.
    3. Compare the two results with tight tolerances.

Usage:
    pytest tests/test_cross_validate.py -v -s
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Paths to the two virtual-environment Python interpreters
# ---------------------------------------------------------------------------
_ENVS = Path("/home/inference/environments")
RAPIDS_PYTHON = str(_ENVS / "scib-rapids" / "bin" / "python")
METRICS_PYTHON = str(_ENVS / "scib-metrics" / "bin" / "python")


def _check_venvs():
    for p in (RAPIDS_PYTHON, METRICS_PYTHON):
        if not Path(p).exists():
            pytest.skip(f"Virtual-env interpreter not found: {p}")


# ---------------------------------------------------------------------------
# Shared test-data fixture (saved to a temp dir as .npy files)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shared_data(tmp_path_factory):
    """Generate deterministic test data and write to disk."""
    _check_venvs()
    d = tmp_path_factory.mktemp("crossval")

    rng = np.random.default_rng(42)
    n_cells, n_features, n_labels, n_batches, n_neighbors = 200, 50, 5, 3, 30

    X = rng.normal(size=(n_cells, n_features)).astype(np.float64)
    labels = rng.integers(0, n_labels, size=(n_cells,))
    batch = rng.integers(0, n_batches, size=(n_cells,))

    # Compute kNN once (deterministic, sklearn)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)

    # Second embedding for PCR comparison
    X_post = rng.normal(size=(n_cells, n_features)).astype(np.float64)

    np.save(d / "X.npy", X)
    np.save(d / "labels.npy", labels)
    np.save(d / "batch.npy", batch)
    np.save(d / "dists.npy", dists)
    np.save(d / "inds.npy", inds)
    np.save(d / "X_post.npy", X_post)

    return d


# ---------------------------------------------------------------------------
# Helper: run a Python snippet in a given venv and return loaded results
# ---------------------------------------------------------------------------
def _run_in_venv(python: str, script: str, timeout: int = 120) -> dict:
    """Run *script* with *python* and return the JSON result dict."""
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed (rc={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout.strip()


def _run_metric(python: str, pkg: str, data_dir: Path, metric_code: str) -> dict:
    """Run metric computation in a venv and load the results from .npy files.

    *metric_code* is a Python snippet that:
      - can use variables: X, labels, batch, dists, inds, X_post, nn, data_dir, np, pkg
      - must assign results to a dict called ``results``
        whose values are scalars or numpy arrays.
      - results are saved as .npy files by the wrapper.
    """
    # The wrapper script loads data, runs the user snippet, and saves results.
    wrapper = textwrap.dedent(f"""\
        import json, sys, warnings
        import numpy as np
        warnings.filterwarnings("ignore")

        data_dir = "{data_dir}"
        pkg = "{pkg}"

        X = np.load(data_dir + "/X.npy")
        labels = np.load(data_dir + "/labels.npy")
        batch = np.load(data_dir + "/batch.npy")
        dists = np.load(data_dir + "/dists.npy")
        inds = np.load(data_dir + "/inds.npy")
        X_post = np.load(data_dir + "/X_post.npy")

        if pkg == "scib_metrics":
            import scib_metrics as lib
            from scib_metrics.nearest_neighbors import NeighborsResults
        else:
            import scib_rapids as lib
            from scib_rapids.nearest_neighbors import NeighborsResults

        nn = NeighborsResults(indices=inds, distances=dists)

        # --- user metric code ---
        {textwrap.indent(metric_code, "        ").strip()}
        # --- end ---

        # Serialize results
        out = {{}}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                path = data_dir + f"/{{pkg}}_{{k}}.npy"
                np.save(path, v)
                out[k] = {{"type": "array", "path": path}}
            else:
                out[k] = {{"type": "scalar", "value": float(v)}}
        print(json.dumps(out))
    """)

    raw = _run_in_venv(python, wrapper)
    meta = json.loads(raw)
    loaded = {}
    for k, info in meta.items():
        if info["type"] == "array":
            loaded[k] = np.load(info["path"])
        else:
            loaded[k] = info["value"]
    return loaded


def _compare(
    rapids_res: dict,
    metrics_res: dict,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    label: str = "",
):
    """Assert all result keys match within tolerance."""
    assert set(rapids_res.keys()) == set(metrics_res.keys()), (
        f"Key mismatch: {rapids_res.keys()} vs {metrics_res.keys()}"
    )
    for k in rapids_res:
        r = rapids_res[k]
        m = metrics_res[k]
        if isinstance(r, np.ndarray):
            np.testing.assert_allclose(
                r, m, atol=atol, rtol=rtol,
                err_msg=f"{label} key={k}",
            )
        else:
            np.testing.assert_allclose(
                r, m, atol=atol, rtol=rtol,
                err_msg=f"{label} key={k}",
            )


# ---------------------------------------------------------------------------
# Individual metric comparison tests
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """Verify numerical agreement between scib-rapids and scib-metrics."""

    # --- Silhouette ---

    def test_silhouette_label(self, shared_data):
        code = """\
result = lib.silhouette_label(X, labels, rescale=True)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="silhouette_label")

    def test_silhouette_label_unscaled(self, shared_data):
        code = """\
result = lib.silhouette_label(X, labels, rescale=False)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="silhouette_label_unscaled")

    def test_silhouette_batch(self, shared_data):
        code = """\
result = lib.silhouette_batch(X, labels, batch)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="silhouette_batch")

    def test_bras(self, shared_data):
        code = """\
result = lib.bras(X, labels, batch)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="bras")

    # --- LISI ---

    def test_lisi_knn(self, shared_data):
        code = """\
result = lib.lisi_knn(nn, labels, perplexity=10)
results = {"lisi": np.asarray(result)}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="lisi_knn")

    def test_ilisi_knn(self, shared_data):
        code = """\
result = lib.ilisi_knn(nn, batch, perplexity=10)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="ilisi_knn")

    def test_clisi_knn(self, shared_data):
        code = """\
result = lib.clisi_knn(nn, labels, perplexity=10)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="clisi_knn")

    # --- kBET ---

    def test_kbet(self, shared_data):
        code = """\
acc, stats, pvals = lib.kbet(nn, batch)
results = {"acc": acc, "stats": np.asarray(stats), "pvals": np.asarray(pvals)}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="kbet")

    def test_kbet_per_label(self, shared_data):
        code = """\
result = lib.kbet_per_label(nn, batch, labels)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="kbet_per_label")

    # --- Graph connectivity ---

    def test_graph_connectivity(self, shared_data):
        code = """\
result = lib.graph_connectivity(nn, labels)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="graph_connectivity")

    # --- Isolated labels ---

    def test_isolated_labels(self, shared_data):
        code = """\
result = lib.isolated_labels(X, labels, batch)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="isolated_labels")

    # --- PCR comparison ---

    def test_pcr_comparison(self, shared_data):
        code = """\
result = lib.pcr_comparison(X, X_post, batch, categorical=True)
results = {"score": result}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="pcr_comparison")

    # --- NMI/ARI (kmeans — note: stochastic, but we fix seeds) ---
    # KMeans uses different RNGs (JAX vs NumPy), so exact match is unlikely.
    # We skip kmeans and only test Leiden which is deterministic given the same graph.

    def test_nmi_ari_leiden(self, shared_data):
        code = """\
result = lib.nmi_ari_cluster_labels_leiden(nn, labels, optimize_resolution=False, resolution=1.0)
results = {"nmi": result["nmi"], "ari": result["ari"]}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="nmi_ari_leiden")

    # --- Utility functions ---

    def test_cdist_euclidean(self, shared_data):
        code = """\
d = lib.utils.cdist(X[:50], X[50:100])
results = {"dists": np.asarray(lib.utils.get_ndarray(d))}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="cdist_euclidean")

    def test_cdist_cosine(self, shared_data):
        code = """\
d = lib.utils.cdist(X[:50], X[50:100], metric="cosine")
results = {"dists": np.asarray(lib.utils.get_ndarray(d))}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="cdist_cosine")

    def test_pca(self, shared_data):
        code = """\
pca_result = lib.utils.pca(X, n_components=10)
results = {
    "variance_ratio": np.asarray(pca_result.variance_ratio),
    "coordinates": np.asarray(np.abs(lib.utils.get_ndarray(pca_result.coordinates))),
}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        # PCA coordinates may differ in sign; compare variance ratios exactly
        # and coordinates by absolute value
        _compare(r, m, atol=1e-5, rtol=1e-5, label="pca")

    def test_silhouette_samples(self, shared_data):
        code = """\
sil = lib.utils.silhouette_samples(X, labels)
results = {"sil": np.asarray(sil)}
"""
        r = _run_metric(RAPIDS_PYTHON, "scib_rapids", shared_data, code)
        m = _run_metric(METRICS_PYTHON, "scib_metrics", shared_data, code)
        _compare(r, m, atol=1e-6, rtol=1e-6, label="silhouette_samples")
