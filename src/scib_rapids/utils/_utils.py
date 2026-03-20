import warnings

import cupy as cp
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array

from scib_rapids._types import ArrayLike, NdArray


def get_ndarray(x: cp.ndarray | np.ndarray) -> np.ndarray:
    """Convert CuPy device array to NumPy array."""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def one_hot(y: NdArray, n_classes: int | None = None) -> cp.ndarray:
    """One-hot encode an array.

    Parameters
    ----------
    y
        Array of shape (n_cells,) or (n_cells, 1).
    n_classes
        Number of classes. If None, inferred from the data.

    Returns
    -------
    one_hot: cp.ndarray
        Array of shape (n_cells, n_classes).
    """
    y = cp.asarray(y).ravel()
    n_classes = n_classes or int(cp.max(y).item()) + 1
    return cp.eye(n_classes, dtype=cp.float32)[y]


def check_square(X: ArrayLike):
    """Check if a matrix is square."""
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix")


def convert_knn_graph_to_idx(X: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Convert a kNN graph to indices and distances."""
    check_array(X, accept_sparse="csr")
    check_square(X)

    n_neighbors = np.unique(X.nonzero()[0], return_counts=True)[1]
    if len(np.unique(n_neighbors)) > 1:
        raise ValueError("Each cell must have the same number of neighbors.")

    n_neighbors = int(np.unique(n_neighbors)[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Precomputed sparse input")
        nn_obj = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(X)
        kneighbors = nn_obj.kneighbors(X)
    return kneighbors
