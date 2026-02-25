import cython
import numpy as np
cimport numpy as cnp
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool

from libc.stdint cimport intptr_t, int32_t, int64_t
from libc.string cimport memcpy
from multipers.simplex_tree_multi import SimplexTreeMulti

cdef extern from "ext_interface/function_delaunay_interface.hpp" namespace "multipers":

  cdef cppclass function_delaunay_interface_input_data "multipers::function_delaunay_interface_input<int>":
    function_delaunay_interface_input_data() except + nogil
    vector[vector[double]] points
    vector[double] function_values

  cdef cppclass function_delaunay_interface_output_data "multipers::function_delaunay_interface_output<int>":
    function_delaunay_interface_output_data() except + nogil
    vector[pair[double,double]] filtration_values
    vector[vector[int]] boundaries
    vector[int] dimensions

  cdef cppclass function_delaunay_simplextree_interface_output_data "multipers::function_delaunay_simplextree_interface_output":
    function_delaunay_simplextree_interface_output_data() except + nogil
    function_delaunay_simplextree_interface_output_data(const function_delaunay_simplextree_interface_output_data&) except + nogil

  bool function_delaunay_interface_available "multipers::function_delaunay_interface_available"() except + nogil

  function_delaunay_interface_output_data function_delaunay_interface "multipers::function_delaunay_interface<int>"(
      const function_delaunay_interface_input_data&,
      int,
      bool,
      bool
  ) except + nogil

  function_delaunay_simplextree_interface_output_data function_delaunay_simplextree_interface "multipers::function_delaunay_simplextree_interface<int>"(
      const function_delaunay_interface_input_data&,
      bool
  ) except + nogil


def _is_available():
    return function_delaunay_interface_available()



# contiguous memory even though slices
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

cdef inline object vect_pair_double_to_array(const vector[pair[double,double]]& filtration_values):
    cdef Py_ssize_t n = filtration_values.size()
    cdef object out = np.empty((n, 2), dtype=np.double)
    cdef double[:, :] out_view = out
    cdef Py_ssize_t i
    for i in range(n):
        out_view[i, 0] = filtration_values[i].first
        out_view[i, 1] = filtration_values[i].second
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def function_delaunay_to_slicer(
    slicer,
    object point_cloud,
    object function_values,
    int degree,
    bint multi_chunk,
    bint verbose,
):
    cdef double[:, ::1] point_cloud_view = np.ascontiguousarray(point_cloud, dtype=np.double)
    cdef double[::1] function_values_view = np.ascontiguousarray(function_values, dtype=np.double)

    num_pts, dim = point_cloud_view.shape[0], point_cloud_view.shape[1]
    if function_values_view.shape[0] != num_pts:
        raise ValueError(f"Got {point_cloud.shape=} and {function_values.shape=}.")

    cdef function_delaunay_interface_input_data interface_input
    cdef function_delaunay_interface_output_data interface_output

    interface_input.points.resize(num_pts)
    for i in range(num_pts):
        interface_input.points[i].resize(dim)
        if dim > 0:
            memcpy(&interface_input.points[i][0], &point_cloud_view[i, 0], dim * sizeof(double))


    interface_input.function_values.resize(num_pts)
    if num_pts > 0:
        memcpy(&interface_input.function_values[0], &function_values_view[0], num_pts * sizeof(double))

    with nogil:
        interface_output = function_delaunay_interface(
            interface_input,
            degree,
            multi_chunk,
            verbose,
        )
    num_generators = interface_output.dimensions.size()

    out_boundaries = vect_vect_boundary_to_numpy_slices(interface_output.boundaries)
    if num_generators == 0:
        out_dimensions = np.empty(0, dtype=np.intc)
    else:
        out_dimensions = np.asarray(
            <int[:num_generators]>(<int*>&interface_output.dimensions[0]),
            dtype=np.intc,
        )
    out_filtrations = vect_pair_double_to_array(interface_output.filtration_values)
    out_filtrations = np.asarray(out_filtrations, dtype=slicer.dtype)
    new_slicer = type(slicer)(out_boundaries, out_dimensions, out_filtrations)
    slicer._from_ptr(new_slicer.get_ptr())
    return slicer


@cython.boundscheck(False)
@cython.wraparound(False)
def function_delaunay_to_simplextree(
    simplextree,
    object point_cloud,
    object function_values,
    bint verbose,
):
    cdef double[:, ::1] point_cloud_view = np.ascontiguousarray(point_cloud, dtype=np.double)
    cdef double[::1] function_values_view = np.ascontiguousarray(function_values, dtype=np.double)
    cdef Py_ssize_t num_pts, dim
    cdef Py_ssize_t i
    cdef intptr_t old_ptr
    cdef function_delaunay_simplextree_interface_output_data* old_cpp_ptr
    cdef function_delaunay_interface_input_data interface_input
    cdef function_delaunay_simplextree_interface_output_data interface_output

    num_pts, dim = point_cloud_view.shape[0], point_cloud_view.shape[1]
    if function_values_view.shape[0] != num_pts:
        raise ValueError(f"Got {point_cloud.shape=} and {function_values.shape=}.")

    interface_input.points.resize(num_pts)
    for i in range(num_pts):
        interface_input.points[i].resize(dim)
        if dim > 0:
            memcpy(&interface_input.points[i][0], &point_cloud_view[i, 0], dim * sizeof(double))

    interface_input.function_values.resize(num_pts)
    if num_pts > 0:
        memcpy(&interface_input.function_values[0], &function_values_view[0], num_pts * sizeof(double))

    with nogil:
        interface_output = function_delaunay_simplextree_interface(
            interface_input,
            verbose,
        )

    out_f64 = SimplexTreeMulti(num_parameters=2, dtype=np.float64, kcritical=False, ftype="Contiguous")
    old_ptr = <intptr_t>out_f64.thisptr
    out_f64.thisptr = <intptr_t>(new function_delaunay_simplextree_interface_output_data(interface_output))
    old_cpp_ptr = <function_delaunay_simplextree_interface_output_data*>old_ptr
    del old_cpp_ptr

    if type(simplextree) is type(out_f64):
        return out_f64
    raise RuntimeError("fixme")

    out = type(simplextree)(out_f64.project_on_line(parameter=0), num_parameters=2)
    lowerstar_values = np.empty(out_f64.num_vertices, dtype=out.dtype)
    for simplex, filtration in out_f64.get_skeleton(0):
        if len(simplex) == 1:
            lowerstar_values[simplex[0]] = filtration[1]
    out.fill_lowerstar(lowerstar_values, 1)
    return out
