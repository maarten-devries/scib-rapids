# scib-rapids

[![PyPI][badge-pypi]][link-pypi]
[![Docs][badge-docs]][link-docs]
[![Build][badge-build]][link-build]
[![Coverage][badge-cov]][link-cov]
[![Discourse][badge-discourse]][link-discourse]
[![Chat][badge-zulip]][link-zulip]

[badge-pypi]: https://img.shields.io/pypi/v/scib-rapids.svg
[link-pypi]: https://pypi.org/project/scib-rapids
[badge-docs]: https://readthedocs.org/projects/scib-rapids/badge/?version=latest
[link-docs]: https://scib-rapids.readthedocs.io/en/latest/?badge=latest
[badge-build]: https://github.com/maarten-devries/scib-rapids/actions/workflows/test.yaml/badge.svg
[link-build]: https://github.com/maarten-devries/scib-rapids/actions/workflows/test.yaml/
[badge-cov]: https://codecov.io/gh/maarten-devries/scib-rapids/branch/main/graph/badge.svg
[link-cov]: https://codecov.io/gh/maarten-devries/scib-rapids
[badge-discourse]: https://img.shields.io/discourse/posts?color=yellow&logo=discourse&server=https%3A%2F%2Fdiscourse.scverse.org
[link-discourse]: https://discourse.scverse.org/
[badge-zulip]: https://img.shields.io/badge/zulip-join_chat-brightgreen.svg
[link-zulip]: https://scverse.zulipchat.com/

GPU-accelerated metrics for benchmarking single-cell integration outputs using RAPIDS (cuML, CuPy).

This package provides the same metrics as [scib-metrics](https://github.com/YosefLab/scib-metrics) but replaces JAX with [RAPIDS](https://rapids.ai/) (CuPy, cuML) for GPU acceleration. All implementations leverage CuPy for device-level computation on NVIDIA GPUs.

## Metrics

- **Silhouette**: `silhouette_label`, `silhouette_batch`, `bras`
- **LISI**: `lisi_knn`, `ilisi_knn`, `clisi_knn`
- **kBET**: `kbet`, `kbet_per_label`
- **Clustering**: `nmi_ari_cluster_labels_kmeans`, `nmi_ari_cluster_labels_leiden`
- **Graph connectivity**: `graph_connectivity`
- **Isolated labels**: `isolated_labels`
- **PCR comparison**: `pcr_comparison`

## Getting started

Please refer to the [documentation][link-docs].

## Installation

You need to have Python 3.11 or newer and a CUDA-capable GPU. We recommend installing in a [conda](https://docs.conda.io/en/latest/miniconda.html) environment with RAPIDS pre-installed.

1. Install the latest release on PyPI:

```bash
pip install scib-rapids
```

2. Install the latest development version:

```bash
pip install git+https://github.com/maarten-devries/scib-rapids.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse Discourse][link-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

If you use `scib-rapids`, please cite the original single-cell integration benchmarking work:

```bibtex
@article{luecken2022benchmarking,
  title={Benchmarking atlas-level data integration in single-cell genomics},
  author={Luecken, Malte D and B{\"u}ttner, Maren and Chaichoompu, Kridsadakorn and Danese, Anna and Interlandi, Marta and M{\"u}ller, Michaela F and Strobl, Daniel C and Zappia, Luke and Dugas, Martin and Colom{\'e}-Tatch{\'e}, Maria and others},
  journal={Nature methods},
  volume={19},
  number={1},
  pages={41--50},
  year={2022},
  publisher={Nature Publishing Group}
}
```

[issue-tracker]: https://github.com/maarten-devries/scib-rapids/issues
[changelog]: https://scib-rapids.readthedocs.io/en/latest/changelog.html
