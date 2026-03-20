import cupy as cp
import numpy as np
import scipy.sparse as sp

NdArray = np.ndarray | cp.ndarray
ArrayLike = np.ndarray | sp.spmatrix | cp.ndarray
