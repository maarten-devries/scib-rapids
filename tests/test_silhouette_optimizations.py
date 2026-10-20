"""Independent references for bounded and reusable GPU silhouettes."""

import importlib

import cupy as cp
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples as cpu_silhouettes

import scib_rapids as scib

silhouette_module = importlib.import_module("scib_rapids.metrics._silhouette")
isolated_module = importlib.import_module("scib_rapids.metrics._isolated_labels")


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize(
    "case", ["ordinary", "singleton", "one_label", "all_singletons", "duplicates", "identical", "large_offset"]
)
def test_silhouettes_match_independent_reference(metric, case):
    """Match CPU silhouettes, including zero-valued undefined cases."""
    if metric == "cosine" and case == "large_offset":
        pytest.skip("Near-parallel cosine distances are below float32 precision")
    X = np.random.default_rng(42).normal(size=(35, 7)).astype(np.float32)
    labels = np.repeat([2, 5, 11], [13, 5, 17])
    if case == "singleton":
        labels[0] = 19
    elif case == "one_label":
        labels[:] = 2
    elif case == "all_singletons":
        labels = np.arange(len(labels))
    elif case == "duplicates":
        X[::3] = X[0]
    elif case == "identical":
        X[:] = X[0]
    elif case == "large_offset":
        X += 10000

    actual = scib.utils.silhouette_samples(X, labels, chunk_size=8, metric=metric)
    if 1 < np.unique(labels).size < len(labels):
        expected = cpu_silhouettes(X.astype(np.float64), labels, metric=metric)
    else:
        expected = np.zeros(len(labels))
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
    if case == "singleton":
        assert actual[0] == 0


@pytest.mark.parametrize("labels", [[], [2], [2, 2, 2], [2, 5, 11]])
def test_degenerate_silhouettes_do_not_call_cuml(monkeypatch, labels):
    """Do not pass label configurations rejected by cuML into its kernel."""

    def unexpected(*args, **kwargs):
        pytest.fail("cuML must not be called for degenerate labels")

    monkeypatch.setattr("cuml.metrics.cluster.silhouette_samples", unexpected)
    actual = scib.utils.silhouette_samples(np.ones((len(labels), 3)), np.asarray(labels))
    np.testing.assert_array_equal(actual, np.zeros(len(labels)))


@pytest.mark.parametrize("chunk_size", [7, 4096])
def test_silhouettes_forward_chunk_size_and_contiguous_float32(monkeypatch, chunk_size):
    """Keep cuML distance tiles bounded and honor explicit chunk sizes."""
    calls = []

    def kernel(X, labels, **kwargs):
        assert X.dtype == cp.float32
        assert X.flags.c_contiguous
        assert labels.dtype == cp.int32
        calls.append(kwargs)
        return cp.zeros(len(X))

    monkeypatch.setattr("cuml.metrics.cluster.silhouette_samples", kernel)
    X = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    scib.utils.silhouette_samples(X, np.array([2, 2, 5, 5]), chunk_size=chunk_size)
    assert calls == [{"metric": "euclidean", "chunksize": chunk_size}]


@pytest.mark.parametrize("mode", ["mean_other", "furthest"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_alternative_silhouettes_preserve_distance_reductions(mode, metric):
    """Keep BRAS reductions, checked independently using SciPy distances."""
    X = np.random.default_rng(7).normal(size=(35, 7)).astype(np.float32)
    labels = np.repeat([2, 5, 11], [13, 5, 17])
    distances = cdist(X, X, metric=metric)
    expected = []
    for i, label in enumerate(labels):
        same = labels == label
        intra = distances[i, same].sum() / (same.sum() - 1)
        if mode == "mean_other":
            inter = distances[i, ~same].mean()
        else:
            inter = max(distances[i, labels == other].mean() for other in np.unique(labels) if other != label)
        expected.append((inter - intra) / max(inter, intra))
    actual = scib.utils.silhouette_samples(X, labels, chunk_size=8, metric=metric, between_cluster_distances=mode)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize("rescale", [False, True])
@pytest.mark.parametrize("batch_counts", [(1, 1, 2), (2, 2, 3), (2, 2, 2)])
def test_precomputed_silhouettes_preserve_weighting_without_recomputation(monkeypatch, rescale, batch_counts):
    """Reuse raw silhouettes without mutating them or changing aggregation."""
    sizes = [8, 2, 4]
    labels = np.repeat(np.arange(3), sizes)
    batches = np.concatenate([np.arange(size) % count for size, count in zip(sizes, batch_counts, strict=True)])
    raw = np.repeat([-0.6, 0.8, 0.2], sizes).astype(np.float32)
    X = np.zeros((len(labels), 2))
    transformed = (raw + 1) / 2 if rescale else raw
    isolated_types = [0, 1] if batch_counts != (2, 2, 2) else [0, 1, 2]
    expected_isolated = np.mean([transformed[labels == label].mean() for label in isolated_types])

    def unexpected(*args, **kwargs):
        pytest.fail("Precomputed silhouettes must not recompute distances")

    monkeypatch.setattr(silhouette_module, "silhouette_samples", unexpected)
    monkeypatch.setattr(isolated_module, "silhouette_samples", unexpected)
    assert scib.silhouette_label(X, labels, rescale=rescale, precomputed_silhouettes=raw) == pytest.approx(
        (raw.mean() + 1) / 2 if rescale else raw.mean()
    )
    assert scib.isolated_labels(X, labels, batches, rescale=rescale, precomputed_silhouettes=raw) == pytest.approx(
        expected_isolated
    )
    np.testing.assert_array_equal(raw, np.repeat([-0.6, 0.8, 0.2], sizes).astype(np.float32))


@pytest.mark.parametrize("shape", [(3,), (4, 1), (1, 4)])
def test_precomputed_silhouettes_require_aligned_vector(shape):
    """Reject malformed reused vectors in both public metrics."""
    X = np.ones((4, 2))
    labels = np.array([2, 2, 5, 5])
    for metric, args in [(scib.silhouette_label, ()), (scib.isolated_labels, (np.array([0, 1, 0, 1]),))]:
        with pytest.raises(ValueError, match="vector aligned"):
            metric(X, labels, *args, precomputed_silhouettes=np.zeros(shape))
        with pytest.raises(ValueError, match="vector aligned"):
            metric(X[:3], labels, *args, precomputed_silhouettes=np.zeros(4))


def test_precomputed_and_computed_scores_match():
    """The optional precomputed path agrees with each metric's regular path."""
    X = np.random.default_rng(11).normal(size=(84, 7)).astype(np.float32)
    labels = np.repeat([2, 5, 11], [48, 24, 12])
    batches = np.arange(len(labels)) % 3
    raw = scib.utils.silhouette_samples(X, labels)
    for metric, args in [(scib.silhouette_label, ()), (scib.isolated_labels, (batches,))]:
        assert metric(X, labels, *args, precomputed_silhouettes=raw) == pytest.approx(metric(X, labels, *args))


@pytest.mark.parametrize("rescale", [False, True])
def test_batch_silhouettes_keep_macro_weighting_and_skip_rules(monkeypatch, rescale):
    """Average per type, not per cell, skipping undefined batch groupings."""
    labels = np.repeat([2, 5, 11, 19], [6, 4, 3, 3])
    batches = np.array([3, 3, 3, 3, 7, 7, 3, 3, 7, 7, 3, 3, 3, 3, 7, 9])
    calls = []

    def silhouettes(X, batch_labels, **kwargs):
        calls.append((X[:, 0].tolist(), batch_labels.tolist()))
        return np.full(len(X), -0.2 if len(X) == 6 else 0.8)

    monkeypatch.setattr(silhouette_module, "silhouette_samples", silhouettes)
    score = scib.silhouette_batch(np.arange(len(labels))[:, None], labels, batches, rescale=rescale)
    assert score == pytest.approx(0.5)
    assert calls == [([0, 1, 2, 3, 4, 5], [3, 3, 3, 3, 7, 7]), ([6, 7, 8, 9], [3, 3, 7, 7])]


@pytest.mark.parametrize("labels,batches", [([], []), ([2], [3]), ([2, 2, 2], [3, 3, 3]), ([2, 2, 2], [3, 7, 9])])
def test_batch_silhouettes_return_nan_when_undefined(labels, batches):
    """An evaluation with no eligible cell types has no defined score."""
    assert np.isnan(scib.silhouette_batch(np.ones((len(labels), 2)), np.asarray(labels), np.asarray(batches)))


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("rescale", [False, True])
def test_batch_silhouettes_match_cpu_reference(metric, rescale):
    """Compare full batch scores with independently grouped sklearn scores."""
    X = np.random.default_rng(7).normal(size=(84, 7)).astype(np.float32)
    labels = np.repeat([2, 5, 11], [48, 24, 12])
    batches = np.arange(len(labels)) % 3
    groups = [
        np.abs(cpu_silhouettes(X[labels == label].astype(np.float64), batches[labels == label], metric=metric))
        for label in np.unique(labels)
    ]
    expected = np.mean([np.mean(1 - values if rescale else values) for values in groups])
    assert scib.silhouette_batch(X, labels, batches, metric=metric, rescale=rescale) == pytest.approx(
        expected, abs=1e-6
    )


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_silhouettes_reject_nonpositive_chunk_sizes(chunk_size):
    """Reject invalid bounds instead of letting a kernel choose an unsafe tile."""
    with pytest.raises(ValueError, match="chunk_size"):
        scib.utils.silhouette_samples(np.ones((4, 2)), np.array([0, 0, 1, 1]), chunk_size=chunk_size)
