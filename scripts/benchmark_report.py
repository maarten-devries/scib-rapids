#!/usr/bin/env python3
"""Benchmark report: compare scib-rapids vs scib-metrics speed and numerical agreement.

Runs each metric in separate subprocesses (one per venv) so that JAX and CuPy
don't interfere.  Outputs a formatted table with timings, speedup, and the
numerical difference between the two backends.

Usage:
    python scripts/benchmark_report.py                 # default 500 cells
    python scripts/benchmark_report.py --n-cells 5000  # larger dataset
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ENVS = Path("/home/inference/environments")
RAPIDS_PYTHON = str(_ENVS / "scib-rapids" / "bin" / "python")
METRICS_PYTHON = str(_ENVS / "scib-metrics" / "bin" / "python")

WARMUP = 1
REPEATS = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    name: str
    rapids_time: float
    metrics_time: float
    rapids_value: float | np.ndarray
    metrics_value: float | np.ndarray
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0

    @property
    def speedup(self) -> float:
        if self.rapids_time == 0:
            return float("inf")
        return self.metrics_time / self.rapids_time


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------
def _run_in_venv(python: str, script: str, timeout: int = 300) -> str:
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed (rc={result.returncode}).\n"
            f"--- stderr ---\n{result.stderr[:2000]}"
        )
    return result.stdout.strip()


def _run_metric_timed(python: str, pkg: str, data_dir: str, code: str) -> tuple[float, dict]:
    """Run metric code in a subprocess and return (min_time, results_dict)."""
    wrapper = textwrap.dedent(f"""\
        import json, time, warnings
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

        def _run():
            {textwrap.indent(code, "            ").strip()}
            return results

        # warmup
        for _ in range({WARMUP}):
            _run()

        # timed repeats
        times = []
        for _ in range({REPEATS}):
            t0 = time.perf_counter()
            results = _run()
            times.append(time.perf_counter() - t0)

        best = min(times)

        out = {{"_time": best}}
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
    t = meta.pop("_time")
    loaded = {}
    for k, info in meta.items():
        if info["type"] == "array":
            loaded[k] = np.load(info["path"])
        else:
            loaded[k] = info["value"]
    return t, loaded


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
METRICS: list[tuple[str, str]] = [
    ("silhouette_label", 'results = {"score": lib.silhouette_label(X, labels)}'),
    ("silhouette_batch", 'results = {"score": lib.silhouette_batch(X, labels, batch)}'),
    ("bras", 'results = {"score": lib.bras(X, labels, batch)}'),
    ("isolated_labels", 'results = {"score": lib.isolated_labels(X, labels, batch)}'),
    ("lisi_knn", 'results = {"lisi": np.asarray(lib.lisi_knn(nn, labels, perplexity=10))}'),
    ("ilisi_knn", 'results = {"score": lib.ilisi_knn(nn, batch, perplexity=10)}'),
    ("clisi_knn", 'results = {"score": lib.clisi_knn(nn, labels, perplexity=10)}'),
    ("kbet", textwrap.dedent("""\
        acc, stats, pvals = lib.kbet(nn, batch)
        results = {"acc": acc, "stats": np.asarray(stats), "pvals": np.asarray(pvals)}""")),
    ("kbet_per_label", 'results = {"score": lib.kbet_per_label(nn, batch, labels)}'),
    ("graph_connectivity", 'results = {"score": lib.graph_connectivity(nn, labels)}'),
    ("pcr_comparison", 'results = {"score": lib.pcr_comparison(X, X_post, batch, categorical=True)}'),
    ("nmi_ari_leiden", textwrap.dedent("""\
        r = lib.nmi_ari_cluster_labels_leiden(nn, labels, optimize_resolution=False, resolution=1.0)
        results = {"nmi": r["nmi"], "ari": r["ari"]}""")),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _compute_diffs(r: dict, m: dict) -> tuple[float, float]:
    """Return (max_abs_diff, max_rel_diff) across all keys."""
    max_abs = 0.0
    max_rel = 0.0
    for k in r:
        rv, mv = np.asarray(r[k], dtype=np.float64), np.asarray(m[k], dtype=np.float64)
        abs_diff = float(np.max(np.abs(rv - mv)))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.where(mv != 0, np.abs((rv - mv) / mv), 0.0)
        rel_max = float(np.max(rel_diff))
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_max)
    return max_abs, max_rel


def _representative_value(d: dict) -> str:
    """Pick a single representative value string from a results dict."""
    for k in ("score", "acc", "nmi"):
        if k in d:
            return f"{d[k]:.6f}"
    # first key
    k0 = next(iter(d))
    v = d[k0]
    if isinstance(v, np.ndarray):
        return f"array(mean={np.mean(v):.6f})"
    return f"{v:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark scib-rapids vs scib-metrics")
    parser.add_argument("--n-cells", type=int, default=500, help="Number of cells")
    parser.add_argument("--n-features", type=int, default=50, help="Number of features")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors for kNN")
    args = parser.parse_args()

    # Check venvs exist
    for p in (RAPIDS_PYTHON, METRICS_PYTHON):
        if not Path(p).exists():
            print(f"ERROR: interpreter not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Generate data
    print(f"Generating test data: {args.n_cells} cells, {args.n_features} features, {args.n_neighbors} neighbors ...")
    rng = np.random.default_rng(42)
    n_labels, n_batches = 5, 3

    X = rng.normal(size=(args.n_cells, args.n_features)).astype(np.float64)
    labels = rng.integers(0, n_labels, size=(args.n_cells,))
    batch = rng.integers(0, n_batches, size=(args.n_cells,))

    nbrs = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)
    X_post = rng.normal(size=X.shape).astype(np.float64)

    tmpdir = tempfile.mkdtemp(prefix="scib_bench_")
    for name, arr in [("X", X), ("labels", labels), ("batch", batch),
                      ("dists", dists), ("inds", inds), ("X_post", X_post)]:
        np.save(f"{tmpdir}/{name}.npy", arr)

    print(f"Data saved to {tmpdir}\n")

    # Run benchmarks
    results: list[MetricResult] = []
    n = len(METRICS)
    for i, (name, code) in enumerate(METRICS, 1):
        print(f"[{i}/{n}] {name} ...", end=" ", flush=True)
        try:
            t_rapids, r_rapids = _run_metric_timed(RAPIDS_PYTHON, "scib_rapids", tmpdir, code)
            t_metrics, r_metrics = _run_metric_timed(METRICS_PYTHON, "scib_metrics", tmpdir, code)
            max_abs, max_rel = _compute_diffs(r_rapids, r_metrics)
            mr = MetricResult(
                name=name,
                rapids_time=t_rapids,
                metrics_time=t_metrics,
                rapids_value=r_rapids,
                metrics_value=r_metrics,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
            )
            results.append(mr)
            print(f"done ({mr.speedup:.1f}x)")
        except Exception as e:
            print(f"FAILED: {e}")

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print("\n")
    print("=" * 100)
    print(f"  BENCHMARK REPORT: scib-rapids vs scib-metrics")
    print(f"  Dataset: {args.n_cells} cells, {args.n_features} features, {args.n_neighbors} neighbors")
    print(f"  Timing: best of {REPEATS} runs after {WARMUP} warmup")
    print("=" * 100)

    # Table header
    hdr = f"{'Metric':<22} {'rapids (s)':>10} {'metrics (s)':>11} {'Speedup':>8} {'Max |Δ|':>10} {'Max |Δ|/|x|':>12} {'rapids val':>16} {'metrics val':>16}"
    print()
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        rv_str = _representative_value(r.rapids_value)
        mv_str = _representative_value(r.metrics_value)
        speedup_str = f"{r.speedup:.1f}x"
        print(
            f"{r.name:<22} {r.rapids_time:>10.4f} {r.metrics_time:>11.4f} {speedup_str:>8} "
            f"{r.max_abs_diff:>10.2e} {r.max_rel_diff:>12.2e} {rv_str:>16} {mv_str:>16}"
        )

    print("-" * len(hdr))

    # Summary
    speedups = [r.speedup for r in results if np.isfinite(r.speedup)]
    if speedups:
        geo_mean = np.exp(np.mean(np.log(speedups)))
        print(f"\nGeometric mean speedup: {geo_mean:.2f}x")

    max_abs_all = max(r.max_abs_diff for r in results)
    max_rel_all = max(r.max_rel_diff for r in results)
    print(f"Worst absolute diff:   {max_abs_all:.2e}")
    print(f"Worst relative diff:   {max_rel_all:.2e}")

    all_close = all(r.max_abs_diff < 1e-3 for r in results)
    if all_close:
        print("\nAll metrics agree within 1e-3 absolute tolerance.")
    else:
        print("\nWARNING: Some metrics differ by more than 1e-3!")
        for r in results:
            if r.max_abs_diff >= 1e-3:
                print(f"  - {r.name}: max |Δ| = {r.max_abs_diff:.2e}")

    print()


if __name__ == "__main__":
    main()
