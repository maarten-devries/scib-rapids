# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 0.1.0 (unreleased)

### Added

- Initial release with GPU-accelerated metrics using CuPy/RAPIDS.
- All metrics from scib-metrics ported: silhouette, LISI, kBET, NMI/ARI, graph connectivity, isolated labels, PCR comparison, BRAS.
- CuPy-based utility functions: cdist, pdist_squareform, PCA, KMeans, silhouette_samples, Simpson index.
- NeighborsResults dataclass with sparse graph properties.
- Unit tests validating correctness against scikit-learn.
- Benchmark test suite comparing runtime against scib-metrics.
