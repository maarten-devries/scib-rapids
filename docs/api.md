# API

## Metrics

Import as:

```
import scib_rapids
scib_rapids.ilisi_knn(...)
```

```{eval-rst}
.. module:: scib_rapids
.. currentmodule:: scib_rapids

.. autosummary::
    :toctree: generated

    isolated_labels
    nmi_ari_cluster_labels_kmeans
    nmi_ari_cluster_labels_leiden
    pcr_comparison
    silhouette_label
    silhouette_batch
    bras
    ilisi_knn
    clisi_knn
    kbet
    kbet_per_label
    graph_connectivity
```

## Utils

```{eval-rst}
.. module:: scib_rapids.utils
.. currentmodule:: scib_rapids

.. autosummary::
    :toctree: generated

    utils.cdist
    utils.pdist_squareform
    utils.silhouette_samples
    utils.KMeans
    utils.pca
    utils.principal_component_regression
    utils.one_hot
    utils.compute_simpson_index
    utils.convert_knn_graph_to_idx
    utils.check_square
    utils.diffusion_nn
```

### Nearest neighbors

```{eval-rst}
.. module:: scib_rapids.nearest_neighbors
.. currentmodule:: scib_rapids

.. autosummary::
    :toctree: generated

    nearest_neighbors.pynndescent
    nearest_neighbors.NeighborsResults
```
