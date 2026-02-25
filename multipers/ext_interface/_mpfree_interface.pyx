import cython
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool

from multipers.ext_interface._helper cimport vect_vect_boundary_to_numpy_slices, vect_pair_double_to_array



cdef extern from "ext_interface/mpfree_interface.hpp" namespace "multipers":

  cdef cppclass mpfree_interface_input_data "multipers::mpfree_interface_input<int>":
    mpfree_interface_input_data() except + nogil
    vector[pair[double,double]] filtration_values
    vector[vector[int]] boundaries
    vector[int] dimensions

  cdef cppclass mpfree_interface_output_data "multipers::mpfree_interface_output<int>":
    mpfree_interface_output_data() except + nogil
    vector[pair[double,double]] filtration_values
    vector[vector[int]] boundaries
    vector[int] dimensions

  bool mpfree_interface_available "multipers::mpfree_interface_available"() except + nogil

  mpfree_interface_output_data mpfree_minpres_interface "multipers::mpfree_minpres_interface<int>"(
      const mpfree_interface_input_data&,
      int,
      bool,
      bool,
      bool,
      bool
  ) except + nogil


def _is_available():
    return mpfree_interface_available()



@cython.boundscheck(False)
@cython.wraparound(False)
def minimal_presentation(slicer, int degree, bint
                         full_resolution=True,
                         bint use_clearing=True,
                         bint use_chunk = True,
                         bint verbose=False):
    if not mpfree_interface_available():
        raise RuntimeError("mpfree in-memory interface is not available.")

    cdef cnp.ndarray[cnp.int32_t, ndim=1] dimensions_np = np.ascontiguousarray(
        slicer.get_dimensions(),
        dtype=np.int32,
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] filtrations_np = np.ascontiguousarray(
        slicer.get_filtrations(view=False),
        dtype=np.float64,
    )
    cdef cnp.int32_t[::1] dimensions_view = dimensions_np
    cdef cnp.float64_t[:, ::1] filtrations_view = filtrations_np
    cdef Py_ssize_t num_generators = dimensions_view.shape[0]
    if filtrations_np.ndim != 2 or filtrations_np.shape[1] != 2:
        raise ValueError(
            "mpfree in-memory interface expects a bifiltration with shape (n, 2)."
        )
    if filtrations_np.shape[0] != num_generators:
        raise ValueError(
            f"Invalid slicer data: got {filtrations_np.shape[0]} filtrations and {num_generators} dimensions."
        )

    cdef vector[pair[double, double]] filtration_values
    cdef vector[int] dimensions
    cdef Py_ssize_t i, j
    cdef Py_ssize_t b_start, b_end

    filtration_values.resize(num_generators)
    dimensions.resize(num_generators)
    for i in range(num_generators):
        dimensions[i] = <int>dimensions_view[i]
        filtration_values[i] = pair[double, double](
            <double>filtrations_view[i, 0],
            <double>filtrations_view[i, 1],
        )

    boundaries_packed = slicer.get_boundaries(packed=True)
    cdef cnp.ndarray[int64_t, ndim=1] boundaries_indptr = np.ascontiguousarray(
        boundaries_packed[0],
        dtype=np.int64,
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1] boundaries_indices = np.ascontiguousarray(
        boundaries_packed[1],
        dtype=np.int32,
    )
    cdef int64_t[::1] boundaries_indptr_view = boundaries_indptr
    cdef cnp.int32_t[::1] boundaries_indices_view = boundaries_indices
    cdef vector[vector[int]] boundaries
    boundaries.resize(num_generators)
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
    for i in range(num_generators):
        b_start = <Py_ssize_t>boundaries_indptr_view[i]
        b_end = <Py_ssize_t>boundaries_indptr_view[i + 1]
        if b_start > b_end:
            raise ValueError(
                f"Invalid packed boundaries: indptr is not non-decreasing at simplex {i}."
            )
        boundaries[i].reserve(b_end - b_start)
        for j in range(b_start, b_end):
            boundaries[i].push_back(<int>boundaries_indices_view[j])

    cdef mpfree_interface_input_data interface_input
    interface_input.filtration_values = filtration_values
    interface_input.boundaries = boundaries
    interface_input.dimensions = dimensions

    cdef mpfree_interface_output_data interface_output
    with nogil:
        interface_output = mpfree_minpres_interface(
            interface_input,
            degree,
            full_resolution,
            use_chunk,
            use_clearing,
            verbose,
        )

    out_boundaries = vect_vect_boundary_to_numpy_slices(interface_output.boundaries)
    out_dimensions = np.asarray(interface_output.dimensions, dtype=np.int32)
    out_filtrations = vect_pair_double_to_array(interface_output.filtration_values)
    out_filtrations = np.asarray(out_filtrations, dtype=slicer.dtype)
    return type(slicer)(out_boundaries, out_dimensions, out_filtrations)
