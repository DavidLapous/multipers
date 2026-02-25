
import cython
import numpy as np
cimport numpy as cnp
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool

from libc.stdint cimport intptr_t, int32_t, int64_t
from libc.string cimport memcpy

cdef inline object vect_pair_double_to_array(const vector[pair[double,double]]& filtration_values):

    cdef Py_ssize_t n = filtration_values.size()
    cdef object out = np.empty((n, 2), dtype=np.double)
    cdef double[:, ::1] out_view = out

    if n == 0:
        return out

    memcpy(&out_view[0,0],
           &filtration_values[0],
           n * sizeof(pair[double,double]))

    return out




cdef inline vector[pair[double,double]] view_2_2Darray(filtration_values_):
    cdef double[:,::1] filtration_values = np.ascontiguousarray(filtration_values_, dtype=np.double)

    cdef Py_ssize_t n = filtration_values.shape[0]
    cdef vector[pair[double,double]] out
    out.resize(n)

    if n == 0:
        return out

    memcpy(&out[0],
           &filtration_values[0,0],
           n * sizeof(pair[double,double]))

    return out



cdef inline tuple vect_vect_boundary_to_numpy_slices(vector[vector[int]]& values):
  cdef Py_ssize_t n = values.size()
  cdef Py_ssize_t i, j, row_size
  cdef Py_ssize_t total_size = 0
  cdef cnp.ndarray[int64_t, ndim=1] indptr = np.empty(n + 1, dtype=np.int64)
  cdef int64_t[:] indptr_view = indptr
  cdef cnp.ndarray[int, ndim=1] data = np.empty(0, dtype=np.intc)
  cdef int[:] data_view
  cdef list out

  indptr_view[0] = 0
  for i in range(n):
    row_size = values[i].size()
    total_size += row_size
    indptr_view[i + 1] = total_size

  data = np.empty(total_size, dtype=np.intc)
  data_view = data

  for i in range(n):
    row_size = values[i].size()
    for j in range(row_size):
      data_view[indptr_view[i] + j] = values[i][j]

  out = [None] * n
  for i in range(n):
    out[i] = data[indptr_view[i] : indptr_view[i + 1]]
  return tuple(out)
