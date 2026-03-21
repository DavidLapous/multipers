import cython
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.string cimport memcpy
from libc.stdint cimport int32_t, int64_t
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool

from multipers._helper cimport view_2_2Darray
from multipers.slicer import is_slicer


cdef extern from "ext_interface/hera_interface.hpp" namespace "multipers":

  cdef cppclass hera_module_presentation_input_data "multipers::hera_module_presentation_input<int>":
    hera_module_presentation_input_data() except + nogil
    vector[pair[double,double]] generator_grades
    vector[pair[double,double]] relation_grades
    vector[vector[int]] relation_components

  cdef cppclass hera_interface_params_data "multipers::hera_interface_params":
    hera_interface_params_data() except + nogil
    double hera_epsilon
    double delta
    int max_depth
    int initialization_depth
    int bound_strategy
    int traverse_strategy
    bool tolerate_max_iter_exceeded
    bool stop_asap

  cdef cppclass hera_interface_result_data "multipers::hera_interface_result":
    hera_interface_result_data() except + nogil
    double distance
    double actual_error
    int actual_max_depth
    int n_hera_calls

  cdef cppclass hera_wasserstein_params_data "multipers::hera_wasserstein_params":
    hera_wasserstein_params_data() except + nogil
    double wasserstein_power
    double internal_p
    double delta

  bool hera_interface_available "multipers::hera_interface_available"() except + nogil

  hera_interface_result_data hera_matching_distance_cpp "multipers::hera_matching_distance<int>"(
      const hera_module_presentation_input_data&,
      const hera_module_presentation_input_data&,
      const hera_interface_params_data&,
  ) except + nogil

  double hera_bottleneck_distance_cpp "multipers::hera_bottleneck_distance"(
      const vector[pair[double,double]]&,
      const vector[pair[double,double]]&,
      double,
  ) except + nogil

  double hera_wasserstein_distance_cpp "multipers::hera_wasserstein_distance"(
      const vector[pair[double,double]]&,
      const vector[pair[double,double]]&,
      const hera_wasserstein_params_data&,
  ) except + nogil


def _is_available():
    return hera_interface_available()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[cnp.float64_t, ndim=2] _diagram_to_array(object diagram):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out
    cdef object array_obj = np.asarray(diagram, dtype=np.float64)

    if array_obj.size == 0 and array_obj.ndim == 1:
        array_obj = np.empty((0, 2), dtype=np.float64)

    out = np.ascontiguousarray(array_obj, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(
            "Hera diagram distances expect arrays of shape (n, 2). "
            f"Got ({out.shape[0]}, {out.shape[1] if out.ndim >= 2 else -1})."
        )
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _flatten_diagrams(object diagrams, bint drop_diagonal=True):
    cdef Py_ssize_t num_diagrams = len(diagrams)
    cdef list arrays = [None] * num_diagrams
    cdef Py_ssize_t i, n_rows, total_rows = 0, offset = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr
    cdef cnp.ndarray[cnp.float64_t, ndim=2] points
    cdef cnp.ndarray[int64_t, ndim=1] indptr = np.empty(num_diagrams + 1, dtype=np.int64)
    cdef int64_t[::1] indptr_view = indptr

    indptr_view[0] = 0
    for i in range(num_diagrams):
        arr = _diagram_to_array(diagrams[i])
        if drop_diagonal and arr.shape[0] > 0:
            arr = np.ascontiguousarray(arr[arr[:, 0] != arr[:, 1]], dtype=np.float64)
        arrays[i] = arr
        total_rows += arr.shape[0]
        indptr_view[i + 1] = total_rows

    points = np.empty((total_rows, 2), dtype=np.float64)
    for i in range(num_diagrams):
        arr = arrays[i]
        n_rows = arr.shape[0]
        if n_rows > 0:
            points[offset : offset + n_rows, :] = arr
        offset += n_rows

    return points, indptr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _bottleneck_distance_from_flat(
    double[:, ::1] left_points,
    int64_t[::1] left_indptr,
    double[:, ::1] right_points,
    int64_t[::1] right_indptr,
    Py_ssize_t idx,
    double delta,
) except * nogil:
    cdef Py_ssize_t left_start = left_indptr[idx]
    cdef Py_ssize_t left_stop = left_indptr[idx + 1]
    cdef Py_ssize_t right_start = right_indptr[idx]
    cdef Py_ssize_t right_stop = right_indptr[idx + 1]
    cdef Py_ssize_t left_size = left_stop - left_start
    cdef Py_ssize_t right_size = right_stop - right_start
    cdef vector[pair[double, double]] left_diagram
    cdef vector[pair[double, double]] right_diagram

    left_diagram.resize(left_size)
    if left_size > 0:
        memcpy(
            &left_diagram[0],
            &left_points[left_start, 0],
            left_size * sizeof(pair[double, double]),
        )

    right_diagram.resize(right_size)
    if right_size > 0:
        memcpy(
            &right_diagram[0],
            &right_points[right_start, 0],
            right_size * sizeof(pair[double, double]),
        )

    return hera_bottleneck_distance_cpp(left_diagram, right_diagram, delta)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef hera_module_presentation_input_data _module_input_from_slicer(object slicer):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] dimensions_np
    cdef cnp.ndarray[cnp.float64_t, ndim=2] filtrations_np
    cdef cnp.ndarray[int64_t, ndim=1] boundaries_indptr_np
    cdef cnp.ndarray[cnp.int32_t, ndim=1] boundaries_indices_np
    cdef int32_t[::1] dimensions_view
    cdef int64_t[::1] boundaries_indptr_view
    cdef int32_t[::1] boundaries_indices_view
    cdef object packed_boundaries
    cdef hera_module_presentation_input_data out
    cdef Py_ssize_t num_generators
    cdef Py_ssize_t row_start, row_end, col_end
    cdef Py_ssize_t rel_idx, comp_idx
    cdef int degree
    cdef int32_t boundary_index
    cdef int64_t boundary_start, boundary_stop

    if not is_slicer(slicer):
        raise ValueError(f"Input has to be a slicer. Got {type(slicer)=}.")
    if slicer.num_parameters != 2:
        raise ValueError(
            f"Hera matching distance only supports 2-parameter slicers. Got {slicer.num_parameters=}."
        )
    if slicer.is_kcritical:
        raise ValueError("Hera matching distance expects a 1-critical minimal presentation slicer.")

    dimensions_np = np.ascontiguousarray(slicer.get_dimensions(), dtype=np.int32)
    dimensions_view = dimensions_np
    num_generators = dimensions_view.shape[0]
    if num_generators > 1 and np.any(
        dimensions_np[1:num_generators] < dimensions_np[0 : num_generators - 1]
    ):
        raise ValueError("Hera matching distance expects dimensions sorted in non-decreasing order.")

    filtrations_np = np.ascontiguousarray(slicer.get_filtrations(view=False), dtype=np.float64)
    if filtrations_np.shape[0] != num_generators:
        raise ValueError(
            f"Invalid slicer data: got {filtrations_np.shape[0]} filtrations and {num_generators} dimensions."
        )
    if filtrations_np.ndim != 2 or filtrations_np.shape[1] != 2:
        raise ValueError(
            "Hera matching distance expects bifiltration values with shape (n, 2). "
            f"Got {(filtrations_np.shape[0], filtrations_np.shape[1])}."
        )

    degree = <int>slicer.minpres_degree
    row_start = <Py_ssize_t>np.searchsorted(dimensions_np, degree)
    row_end = <Py_ssize_t>np.searchsorted(dimensions_np, degree + 1)
    col_end = <Py_ssize_t>np.searchsorted(dimensions_np, degree + 2)

    out.generator_grades = view_2_2Darray(filtrations_np[row_start:row_end])
    out.relation_grades = view_2_2Darray(filtrations_np[row_end:col_end])
    out.relation_components.resize(col_end - row_end)

    packed_boundaries = slicer.get_boundaries(packed=True)
    boundaries_indptr_np = np.ascontiguousarray(packed_boundaries[0], dtype=np.int64)
    boundaries_indices_np = np.ascontiguousarray(packed_boundaries[1], dtype=np.int32)
    boundaries_indptr_view = boundaries_indptr_np
    boundaries_indices_view = boundaries_indices_np

    if boundaries_indptr_view.shape[0] != num_generators + 1:
        raise ValueError(
            f"Invalid packed boundaries: got {boundaries_indptr_view.shape[0] - 1} entries and {num_generators} dimensions."
        )
    if boundaries_indptr_view[0] != 0:
        raise ValueError("Invalid packed boundaries: indptr[0] must be 0.")
    if boundaries_indptr_view[num_generators] != boundaries_indices_view.shape[0]:
        raise ValueError(
            f"Invalid packed boundaries: indptr[-1]={boundaries_indptr_view[num_generators]} and indices size={boundaries_indices_view.shape[0]} differ."
        )

    for rel_idx in range(row_end, col_end):
        boundary_start = boundaries_indptr_view[rel_idx]
        boundary_stop = boundaries_indptr_view[rel_idx + 1]
        out.relation_components[rel_idx - row_end].reserve(boundary_stop - boundary_start)
        for comp_idx in range(boundary_start, boundary_stop):
            boundary_index = boundaries_indices_view[comp_idx]
            if boundary_index < row_start or boundary_index >= row_end:
                raise ValueError(
                    "Invalid minimal presentation slicer: relation boundaries must reference degree-d generators only."
                )
            out.relation_components[rel_idx - row_end].push_back(boundary_index - row_start)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def matching_distance(
    object left,
    object right,
    double hera_epsilon=0.001,
    double delta=0.1,
    int max_depth=8,
    int initialization_depth=2,
    int bound_strategy=4,
    int traverse_strategy=1,
    bint tolerate_max_iter_exceeded=False,
    bint stop_asap=True,
    bint return_stats=False,
):
    for name, slicer in (("left", left), ("right", right)):
        if slicer.num_parameters != 2:
            raise ValueError(
                "Matching distance only supports 2-parameter slicers. "
                f"{name} has {slicer.num_parameters=}."
            )
        if slicer.is_kcritical:
            raise ValueError(
                "Matching distance expects 1-critical minimal-presentation slicers."
            )
    cdef hera_module_presentation_input_data left_input = _module_input_from_slicer(left)
    cdef hera_module_presentation_input_data right_input = _module_input_from_slicer(right)
    cdef hera_interface_params_data params
    cdef hera_interface_result_data result

    params.hera_epsilon = hera_epsilon
    params.delta = delta
    params.max_depth = max_depth
    params.initialization_depth = initialization_depth
    params.bound_strategy = bound_strategy
    params.traverse_strategy = traverse_strategy
    params.tolerate_max_iter_exceeded = tolerate_max_iter_exceeded
    params.stop_asap = stop_asap

    with nogil:
        result = hera_matching_distance_cpp(left_input, right_input, params)

    if return_stats:
        return result.distance, {
            "actual_error": result.actual_error,
            "actual_max_depth": result.actual_max_depth,
            "n_hera_calls": result.n_hera_calls,
        }
    return result.distance


@cython.boundscheck(False)
@cython.wraparound(False)
def bottleneck_distance(object left, object right, double delta=0.01):
    cdef vector[pair[double, double]] left_input = view_2_2Darray(_diagram_to_array(left))
    cdef vector[pair[double, double]] right_input = view_2_2Darray(_diagram_to_array(right))
    cdef double out

    with nogil:
        out = hera_bottleneck_distance_cpp(left_input, right_input, delta)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def bottleneck_distances(object left_diagrams, object right_diagrams, double delta=0.01):
    cdef tuple left_flat
    cdef tuple right_flat
    cdef cnp.ndarray[cnp.float64_t, ndim=2] left_points_np
    cdef cnp.ndarray[cnp.float64_t, ndim=2] right_points_np
    cdef cnp.ndarray[int64_t, ndim=1] left_indptr_np
    cdef cnp.ndarray[int64_t, ndim=1] right_indptr_np
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_np
    cdef double[:, ::1] left_points
    cdef double[:, ::1] right_points
    cdef int64_t[::1] left_indptr
    cdef int64_t[::1] right_indptr
    cdef double[::1] out
    cdef Py_ssize_t num_diagrams, i

    if len(left_diagrams) != len(right_diagrams):
        raise ValueError(
            "Left and right diagram batches must contain the same number of diagrams."
        )

    left_flat = _flatten_diagrams(left_diagrams, drop_diagonal=True)
    right_flat = _flatten_diagrams(right_diagrams, drop_diagonal=True)
    left_points_np = left_flat[0]
    left_indptr_np = left_flat[1]
    right_points_np = right_flat[0]
    right_indptr_np = right_flat[1]

    num_diagrams = left_indptr_np.shape[0] - 1
    out_np = np.empty(num_diagrams, dtype=np.float64)
    if num_diagrams == 0:
        return out_np

    left_points = left_points_np
    left_indptr = left_indptr_np
    right_points = right_points_np
    right_indptr = right_indptr_np
    out = out_np

    with nogil:
        for i in prange(num_diagrams, schedule="guided"):
            out[i] = _bottleneck_distance_from_flat(
                left_points,
                left_indptr,
                right_points,
                right_indptr,
                i,
                delta,
            )

    return out_np


@cython.boundscheck(False)
@cython.wraparound(False)
def wasserstein_distance(
    object left,
    object right,
    double order=1.0,
    double internal_p=np.inf,
    double delta=0.01,
):
    cdef vector[pair[double, double]] left_input = view_2_2Darray(_diagram_to_array(left))
    cdef vector[pair[double, double]] right_input = view_2_2Darray(_diagram_to_array(right))
    cdef hera_wasserstein_params_data params
    cdef double out

    params.wasserstein_power = order
    params.internal_p = internal_p
    params.delta = delta

    with nogil:
        out = hera_wasserstein_distance_cpp(left_input, right_input, params)
    return out
