import numpy as np
from pynndescent import NNDescent

from ._dataclass import NeighborsResults


def pynndescent(X: np.ndarray, n_neighbors: int = 30) -> NeighborsResults:
    """Compute nearest neighbors using PyNNDescent.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    n_neighbors
        Number of nearest neighbors.

    Returns
    -------
    NeighborsResults
    """
    index = NNDescent(X, n_neighbors=n_neighbors)
    indices, distances = index.neighbor_graph
    return NeighborsResults(indices=indices, distances=distances)
