import warnings

from scib_rapids._types import NdArray
from scib_rapids.utils import principal_component_regression


def pcr_comparison(
    X_pre: NdArray,
    X_post: NdArray,
    covariate: NdArray,
    scale: bool = True,
    **kwargs,
) -> float:
    """Principal component regression (PCR) comparison.

    Parameters
    ----------
    X_pre
        Pre-integration array of shape (n_cells, n_features).
    X_post
        Post-integration array of shape (n_cells, n_features).
    covariate
        Array of shape (n_cells,) or (n_cells, 1) representing batch/covariate values.
    scale
        Whether to scale the score between 0 and 1.
    kwargs
        Keyword arguments passed into principal_component_regression.

    Returns
    -------
    pcr_compared: float
    """
    if X_pre.shape[0] != X_post.shape[0]:
        raise ValueError("Dimension mismatch: `X_pre` and `X_post` must have the same number of samples.")
    if covariate.shape[0] != X_pre.shape[0]:
        raise ValueError("Dimension mismatch: `X_pre` and `covariate` must have the same number of samples.")

    pcr_pre = principal_component_regression(X_pre, covariate, **kwargs)
    pcr_post = principal_component_regression(X_post, covariate, **kwargs)

    if scale:
        pcr_compared = (pcr_pre - pcr_post) / pcr_pre
        if pcr_compared < 0:
            warnings.warn(
                "PCR comparison score is negative, meaning variance contribution "
                "increased after integration. Setting to 0."
            )
            pcr_compared = 0
    else:
        pcr_compared = pcr_post - pcr_pre

    return pcr_compared
