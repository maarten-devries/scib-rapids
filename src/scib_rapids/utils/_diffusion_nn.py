import logging
from typing import Literal

import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse

from scib_rapids import nearest_neighbors

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _compute_transitions(X: csr_matrix, density_normalize: bool = True):
    """Density-normalized transitions. Adapted from scanpy."""
    if density_normalize:
        q = np.asarray(X.sum(axis=0))
        if not issparse(X):
            Q = np.diag(1.0 / q)
        else:
            Q = scipy.sparse.spdiags(1.0 / q, 0, X.shape[0], X.shape[0])
        K = Q @ X @ Q
    else:
        K = X

    z = np.sqrt(np.asarray(K.sum(axis=0)))
    if not issparse(K):
        Z = np.diag(1.0 / z)
    else:
        Z = scipy.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
    transitions_sym = Z @ K @ Z

    return transitions_sym


def _compute_eigen(
    transitions_sym: csr_matrix,
    n_comps: int = 15,
    sort: Literal["decrease", "increase"] = "decrease",
):
    """Compute eigen decomposition of transition matrix."""
    matrix = transitions_sym
    if n_comps == 0:
        evals, evecs = scipy.linalg.eigh(matrix)
    else:
        n_comps = min(matrix.shape[0] - 1, n_comps)
        ncv = None
        which = "LM" if sort == "decrease" else "SM"
        matrix = matrix.astype(np.float64)
        evals, evecs = scipy.sparse.linalg.eigsh(matrix, k=n_comps, which=which, ncv=ncv)
        evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)
    if sort == "decrease":
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

    return evals, evecs


def diffusion_nn(X: csr_matrix, k: int, n_comps: int = 100) -> "nearest_neighbors.NeighborsResults":
    """Diffusion-based neighbors.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_cells) with non-zero values representing connectivities.
    k
        Number of nearest neighbours to select.
    n_comps
        Number of components for diffusion map.

    Returns
    -------
    Neighbors results
    """
    transitions = _compute_transitions(X)
    evals, evecs = _compute_eigen(transitions, n_comps=n_comps)
    evals += _EPS
    embedding = evecs
    scaled_evals = np.array([e if e == 1 else e / (1 - e) for e in evals])
    embedding *= scaled_evals
    nn_result = nearest_neighbors.pynndescent(embedding, n_neighbors=k + 1)

    return nn_result
