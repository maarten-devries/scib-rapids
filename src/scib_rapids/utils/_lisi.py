import cupy as cp
import numpy as np

from ._utils import get_ndarray

NdArray = np.ndarray | cp.ndarray

# Fused CUDA kernel: binary search + Simpson index per cell.
# Each thread handles one cell's entire computation.
_SIMPSON_KERNEL_CODE = r"""
extern "C" __global__
void compute_simpson(
    const float* __restrict__ knn_dists,   // (n_cells, n_neighbors)
    const int*   __restrict__ knn_labels,  // (n_cells, n_neighbors)
    const char*  __restrict__ self_mask,   // (n_cells, n_neighbors)
    float*       __restrict__ out,         // (n_cells,)
    const int    n_cells,
    const int    n_neighbors,
    const int    n_labels,
    const float  logU,
    const float  tol,
    const int    max_iter
) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= n_cells) return;

    int offset = cell * n_neighbors;

    // Binary search for beta that gives target perplexity
    float beta = 1.0f;
    float betamin = -1e30f;  // use large finite values instead of INFINITY
    float betamax = 1e30f;

    float sumP = 0.0f;
    float sumDP = 0.0f;

    // Initial Hbeta computation
    for (int j = 0; j < n_neighbors; j++) {
        if (self_mask[offset + j]) {
            float p = expf(-knn_dists[offset + j] * beta);
            sumP += p;
            sumDP += knn_dists[offset + j] * p;
        }
    }
    if (sumP == 0.0f) {
        out[cell] = -1.0f;
        return;
    }

    float H = logf(sumP) + beta * sumDP / sumP;
    float Hdiff = H - logU;

    // Use large-but-finite sentinel to detect "unset" bounds
    // (matches the behavior of using +/-inf in the Python version)
    float BOUND_SENTINEL = 1e30f;
    betamin = -BOUND_SENTINEL;
    betamax = BOUND_SENTINEL;

    for (int iter = 0; iter < max_iter; iter++) {
        if (fabsf(Hdiff) < tol) break;

        if (Hdiff > 0.0f) {
            betamin = beta;
            beta = (betamax >= BOUND_SENTINEL) ? beta * 2.0f : (beta + betamax) * 0.5f;
        } else {
            betamax = beta;
            beta = (betamin <= -BOUND_SENTINEL) ? beta * 0.5f : (beta + betamin) * 0.5f;
        }

        // Recompute Hbeta
        sumP = 0.0f;
        sumDP = 0.0f;
        for (int j = 0; j < n_neighbors; j++) {
            if (self_mask[offset + j]) {
                float p = expf(-knn_dists[offset + j] * beta);
                sumP += p;
                sumDP += knn_dists[offset + j] * p;
            }
        }
        if (sumP == 0.0f) {
            out[cell] = -1.0f;
            return;
        }
        H = logf(sumP) + beta * sumDP / sumP;
        Hdiff = H - logU;
    }

    if (H == 0.0f) {
        out[cell] = -1.0f;
        return;
    }

    // Compute normalized P and accumulate per-label sums in shared memory
    // Use dynamically-indexed shared memory for label_sums
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    float* label_sums = shared_mem + tid * n_labels;

    for (int l = 0; l < n_labels; l++) {
        label_sums[l] = 0.0f;
    }

    for (int j = 0; j < n_neighbors; j++) {
        if (self_mask[offset + j]) {
            float p = expf(-knn_dists[offset + j] * beta) / sumP;
            label_sums[knn_labels[offset + j]] += p;
        }
    }

    // Simpson = sum of squared label probabilities
    float simpson = 0.0f;
    for (int l = 0; l < n_labels; l++) {
        simpson += label_sums[l] * label_sums[l];
    }
    out[cell] = simpson;
}
"""

_SIMPSON_KERNEL = cp.RawKernel(_SIMPSON_KERNEL_CODE, "compute_simpson")


def compute_simpson_index(
    knn_dists: NdArray,
    knn_idx: NdArray,
    row_idx: NdArray,
    labels: NdArray,
    n_labels: int,
    perplexity: float = 30,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute the Simpson index for each cell using a fused CUDA kernel.

    The entire binary search and Simpson computation runs in a single
    GPU kernel launch — no Python loops, no per-cell overhead.

    Parameters
    ----------
    knn_dists
        KNN distances of size (n_cells, n_neighbors).
    knn_idx
        KNN indices of size (n_cells, n_neighbors).
    row_idx
        Idx of each row (n_cells, 1).
    labels
        Cell labels of size (n_cells,).
    n_labels
        Number of labels.
    perplexity
        Measure of the effective number of neighbors.
    tol
        Tolerance for binary search.

    Returns
    -------
    simpson_index
        Simpson index of size (n_cells,).
    """
    knn_dists = cp.asarray(knn_dists, dtype=cp.float32)
    knn_idx = cp.asarray(knn_idx)
    labels = cp.asarray(labels, dtype=cp.int32)
    row_idx = cp.asarray(row_idx)

    n_cells, n_neighbors = knn_dists.shape
    knn_labels = cp.asarray(labels[knn_idx], dtype=cp.int32)
    self_mask = cp.asarray(knn_idx != row_idx, dtype=cp.int8)

    out = cp.empty(n_cells, dtype=cp.float32)
    logU = np.float32(np.log(perplexity))

    # Shared memory: each thread needs n_labels floats for label_sums.
    # Cap block_size so we stay within the 48 KB shared-memory limit.
    MAX_SHARED_BYTES = 49152
    bytes_per_thread = n_labels * 4  # 4 bytes per float
    block_size = min(256, MAX_SHARED_BYTES // bytes_per_thread)
    if block_size < 1:
        block_size = 1
    grid_size = (n_cells + block_size - 1) // block_size
    shared_mem_bytes = block_size * bytes_per_thread

    _SIMPSON_KERNEL(
        (grid_size,),
        (block_size,),
        (
            knn_dists,
            knn_labels,
            self_mask,
            out,
            np.int32(n_cells),
            np.int32(n_neighbors),
            np.int32(n_labels),
            np.float32(logU),
            np.float32(tol),
            np.int32(50),
        ),
        shared_mem=shared_mem_bytes,
    )

    return get_ndarray(out)
