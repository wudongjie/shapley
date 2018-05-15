from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t


@cython.boundscheck(False)
def gini(np.ndarray[DTYPE_t, ndim=1, mode='c'] x):
    """Calculate the Gini coefficient of a numpy array."""
    cdef int n
    cdef np.ndarray[np.float_t, ndim=1, mode='c'] cumx
    n = x.shape[0]
    cumx = np.cumsum(np.sort(x), dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


@cython.boundscheck(False)
@cython.wraparound(False)
def mean_c(np.ndarray[DTYPE_t, ndim=2, mode='c'] data, np.ndarray[np.uint8_t, ndim=1, cast=True] mask):
    cdef int N = data.shape[0]
    cdef int Nc = data.shape[1]
    cdef int x, y
    cdef double m
    for x in range(Nc):
        m = 0.0
        if mask[x]:
            for y in range(N):
                m += data[y, x]
            m = m / N  # mean value
            for y in range(N):
                data[y, x] = m
#    return data
