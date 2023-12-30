import cython
from cython import Py_ssize_t

import numpy as np

from numpy cimport (
    float64_t,
    intp_t,
)

@cython.wraparound(False)
@cython.boundscheck(False)
def ohlcv_grouper(float64_t[:, ::1] out,
               float64_t[:, ::1] values,
               const intp_t[:] labels) -> None:

    cdef:
        Py_ssize_t i, N, K, lab

    if len(labels) == 0:
        return

    N, K = (<object>values).shape

    if out.shape[1] != 5:
        raise ValueError('out array must have 5 columns')
    
    out[:] = np.nan


    with nogil:
        for i in range(N):

            lab = labels[i]
            if lab == -1:
                continue

            # Open column
            if out[lab, 0] != out[lab, 0]:
                out[lab, 0] = values[i, 0]

            # High column
            if out[lab, 1] != out[lab, 1]:
                out[lab, 1] = values[i, 1]
            elif values[i, 1] == values[i, 1]:
                out[lab, 1] = max(out[lab, 1], values[i, 1])

            # Low column
            if out[lab, 2] != out[lab, 2]:
                out[lab, 2] = values[i, 2]
            elif values[i, 2] == values[i, 2]:
                out[lab, 2] = min(out[lab, 2], values[i, 2])

            # Close Column
            if values[i, 3] == values[i, 3]:
                out[lab, 3] = values[i, 3]

            # Volume column
            if out[lab, 4] != out[lab, 4]:
                out[lab, 4] = values[i, 4]
            elif values[i, 4] == values[i, 4]:
                out[lab, 4] = out[lab, 4] + values[i, 4]
