from typing import Literal

import cupy as cp
import numpy as np
from sklearn.utils import check_array

from ._dist import cdist, cdist_sq
from ._utils import get_ndarray


def _tolerance(X: np.ndarray, tol: float) -> float:
    """Return a tolerance which is dependent on the dataset."""
    variances = np.var(X, axis=0)
    return np.mean(variances) * tol


class KMeans:
    """CuPy/RAPIDS implementation of KMeans clustering.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    init
        Cluster centroid initialization method: 'k-means++' or 'random'.
    n_init
        Number of times the k-means algorithm will be initialized.
    max_iter
        Maximum number of iterations.
    tol
        Relative tolerance with regards to inertia to declare convergence.
    seed
        Random seed.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 1,
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int = 0,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol_scale = tol
        self.seed = seed
        if init not in ["k-means++", "random"]:
            raise ValueError("Invalid init method, must be one of ['k-means++' or 'random'].")
        self.init = init

    def _initialize_random(self, X: cp.ndarray, rng: np.random.Generator) -> cp.ndarray:
        n_obs = X.shape[0]
        indices = rng.choice(n_obs, self.n_clusters, replace=False)
        return X[indices].copy()

    def _initialize_plus_plus(self, X: cp.ndarray, rng: np.random.Generator) -> cp.ndarray:
        n_obs = X.shape[0]
        idx = rng.integers(0, n_obs)
        centroids = [X[idx]]
        min_dist_sq = cp.sum((X - centroids[0]) ** 2, axis=1)

        for _ in range(1, self.n_clusters):
            probs = get_ndarray(min_dist_sq / cp.sum(min_dist_sq))
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            n_local_trials = 2 + int(np.log(self.n_clusters))
            candidates_idx = rng.choice(n_obs, n_local_trials, replace=False, p=probs)
            candidates = X[candidates_idx]

            # Compute distances for each candidate
            dist_sq_candidates = cdist_sq(candidates, X)
            dist_sq_candidates = cp.minimum(min_dist_sq[None, :], dist_sq_candidates)
            candidates_pot = cp.sum(dist_sq_candidates, axis=1)

            best = int(cp.argmin(candidates_pot).item())
            min_dist_sq = dist_sq_candidates[best]
            centroids.append(X[candidates_idx[best]])

        return cp.stack(centroids)

    def fit(self, X: np.ndarray):
        """Fit the model to the data."""
        X = check_array(X, dtype=np.float32, order="C")
        self.tol = _tolerance(X, self.tol_scale)
        mean = X.mean(axis=0)
        X_centered = X - mean
        X_gpu = cp.asarray(X_centered)

        best_centroids = None
        best_inertia = np.inf

        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_init):
            centroids, inertia = self._kmeans_full_run(X_gpu, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids

        self.cluster_centroids_ = get_ndarray(best_centroids) + mean
        self.inertia_ = best_inertia

        # Compute final labels
        dist = cdist_sq(X_gpu, best_centroids)
        self.labels_ = get_ndarray(cp.argmin(dist, axis=1))
        return self

    def _kmeans_full_run(self, X: cp.ndarray, rng: np.random.Generator) -> tuple[cp.ndarray, float]:
        if self.init == "k-means++":
            centroids = self._initialize_plus_plus(X, rng)
        else:
            centroids = self._initialize_random(X, rng)

        old_inertia = np.inf
        for _ in range(self.max_iter):
            # Assign labels
            dist = cdist_sq(X, centroids)
            labels = cp.argmin(dist, axis=1)

            # Update centroids
            new_centroids = cp.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                count = cp.sum(mask)
                if count > 0:
                    new_centroids[k] = cp.sum(X[mask], axis=0) / count
                else:
                    new_centroids[k] = centroids[k]

            new_inertia = float(cp.sum(cp.min(dist, axis=1)).item())

            if abs(old_inertia - new_inertia) <= self.tol:
                centroids = new_centroids
                break
            old_inertia = new_inertia
            centroids = new_centroids

        # Final inertia
        dist = cdist_sq(X, centroids)
        final_inertia = float(cp.sum(cp.min(dist, axis=1)).item())
        return centroids, final_inertia
