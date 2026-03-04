import cython
import numpy as np
cimport numpy as cnp
from libc.stdint cimport intptr_t, int64_t
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference

from multipers._helper cimport vect_vect_boundary_to_numpy_slices, vect_pair_double_to_array
import multipers.slicer as mps
from multipers.slicer cimport C_ContiguousSlicer_Matrix0_f64, C_KContiguousSlicer_Matrix0_f64


cdef extern from "ext_interface/multi_critical_interface.hpp" namespace "multipers":

  cdef cppclass multi_critical_interface_input_data "multipers::multi_critical_interface_input<int>":
    multi_critical_interface_input_data() except + nogil
    vector[vector[pair[double,double]]] filtration_values
    vector[vector[int]] boundaries
    vector[int] dimensions

  cdef cppclass multi_critical_interface_output_data "multipers::multi_critical_interface_output<int>":
    multi_critical_interface_output_data() except + nogil
    multi_critical_interface_output_data(const multi_critical_interface_output_data&) except + nogil
    vector[pair[double,double]] filtration_values
    vector[vector[int]] boundaries
    vector[int] dimensions

  bool multi_critical_interface_available "multipers::multi_critical_interface_available"() except + nogil

  multi_critical_interface_output_data multi_critical_resolution_interface "multipers::multi_critical_resolution_interface<int>"(
      const multi_critical_interface_input_data&,
      bool,
      bool,
      bool
  ) except + nogil

  C_ContiguousSlicer_Matrix0_f64 multi_critical_resolution_contiguous_interface_cpp "multipers::multi_critical_resolution_contiguous_interface"(
      C_KContiguousSlicer_Matrix0_f64&,
      bool,
      bool,
      bool
  ) except + nogil

  multi_critical_interface_output_data multi_critical_minpres_interface "multipers::multi_critical_minpres_interface<int>"(
      const multi_critical_interface_input_data&,
      int,
      bool,
      bool,
      bool,
      bool
  ) except + nogil

  vector[multi_critical_interface_output_data] multi_critical_minpres_all_interface "multipers::multi_critical_minpres_all_interface<int>"(
      const multi_critical_interface_input_data&,
      bool,
      bool,
      bool,
      bool
  ) except + nogil


def _is_available():
    return multi_critical_interface_available()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _slicer_from_multi_critical_output(
    object slicer_type,
    multi_critical_interface_output_data& interface_output,
    bint mark_minpres=False,
    int degree=-1,
):
    out_boundaries = vect_vect_boundary_to_numpy_slices(interface_output.boundaries)
    out_dimensions = np.array(interface_output.dimensions, dtype=np.int32) 
    out_filtrations = vect_pair_double_to_array(interface_output.filtration_values)
    out = slicer_type(out_boundaries, out_dimensions, out_filtrations)
    if mark_minpres:
        out.minpres_degree = degree
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef multi_critical_interface_input_data _multi_critical_input_from_slicer(object slicer):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] dimensions_np = np.ascontiguousarray(
        slicer.get_dimensions(),
        dtype=np.int32,
    )
    cdef cnp.int32_t[::1] dimensions_view = dimensions_np
    cdef object packed_boundaries = slicer.get_boundaries(packed=True)
    cdef cnp.ndarray[int64_t, ndim=1] boundaries_indptr_np = np.ascontiguousarray(
        packed_boundaries[0],
        dtype=np.int64,
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1] boundaries_indices_np = np.ascontiguousarray(
        packed_boundaries[1],
        dtype=np.int32,
    )
    cdef int64_t[::1] boundaries_indptr_view = boundaries_indptr_np
    cdef cnp.int32_t[::1] boundaries_indices_view = boundaries_indices_np
    cdef object packed_filtrations
    cdef object filtrations_py
    cdef object indptr_np
    cdef object grades_np
    cdef int64_t[::1] indptr_view
    cdef cnp.float64_t[:, ::1] grades_view
    cdef cnp.float64_t[::1] grade_row_view
    cdef cnp.float64_t[:, ::1] grades_view_2d
    cdef Py_ssize_t num_generators = dimensions_view.shape[0]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t start_idx, end_idx
    cdef Py_ssize_t b_start_idx, b_end_idx
    cdef multi_critical_interface_input_data interface_input

    if boundaries_indptr_view.shape[0] != num_generators + 1:
        raise ValueError(
            f"Invalid packed boundaries: got {boundaries_indptr_view.shape[0] - 1} entries and {num_generators} dimensions."
        )
    if boundaries_indptr_view[0] != 0:
        raise ValueError("Invalid packed boundaries: indptr[0] must be 0.")
    if boundaries_indptr_view[num_generators] != boundaries_indices_view.shape[0]:
        raise ValueError(
            f"Invalid packed boundaries: indptr[-1]={boundaries_indptr_view[num_generators]} "
            f"and indices size={boundaries_indices_view.shape[0]} differ."
        )

    if slicer.is_kcritical:
        packed_filtrations = slicer.get_filtrations(packed=True)
        indptr_np = np.ascontiguousarray(packed_filtrations[0], dtype=np.int64)
        grades_np = np.ascontiguousarray(packed_filtrations[1], dtype=np.float64)
        indptr_view = indptr_np
        if indptr_view.shape[0] != num_generators + 1:
            raise ValueError(
                f"Invalid packed filtrations: got {indptr_view.shape[0] - 1} entries and {num_generators} dimensions."
            )
        if grades_np.ndim != 2 or grades_np.shape[1] != 2:
            raise ValueError(
                f"multi_critical in-memory interface expects packed grades with shape (n, 2). Got {grades_np.shape}."
            )
        grades_view = grades_np
        if indptr_view[0] != 0:
            raise ValueError("Invalid packed filtrations: indptr[0] must be 0.")
        if indptr_view[num_generators] != grades_view.shape[0]:
            raise ValueError(
                f"Invalid packed filtrations: indptr[-1]={indptr_view[num_generators]} and grades rows={grades_view.shape[0]} differ."
            )
    else:
        filtrations_py = slicer.get_filtrations(view=False)
        if len(filtrations_py) != num_generators:
            raise ValueError(
                f"Invalid slicer data: got {len(filtrations_py)} filtration entries and {num_generators} dimensions."
            )

    interface_input.dimensions.resize(num_generators)
    interface_input.boundaries.resize(num_generators)
    interface_input.filtration_values.resize(num_generators)

    for i in range(num_generators):
        interface_input.dimensions[i] = <int>dimensions_view[i]

        b_start_idx = <Py_ssize_t>boundaries_indptr_view[i]
        b_end_idx = <Py_ssize_t>boundaries_indptr_view[i + 1]
        if b_start_idx > b_end_idx:
            raise ValueError(
                f"Invalid packed boundaries: indptr is not non-decreasing at simplex {i}."
            )
        interface_input.boundaries[i].reserve(b_end_idx - b_start_idx)
        for j in range(b_start_idx, b_end_idx):
            interface_input.boundaries[i].push_back(<int>boundaries_indices_view[j])

        if slicer.is_kcritical:
            start_idx = <Py_ssize_t>indptr_view[i]
            end_idx = <Py_ssize_t>indptr_view[i + 1]
            if start_idx > end_idx:
                raise ValueError(
                    f"Invalid packed filtrations: indptr is not non-decreasing at simplex {i}."
                )
            interface_input.filtration_values[i].reserve(end_idx - start_idx)
            for j in range(start_idx, end_idx):
                interface_input.filtration_values[i].push_back(
                    pair[double, double](grades_view[j, 0], grades_view[j, 1])
                )
        else:
            grades = np.ascontiguousarray(filtrations_py[i], dtype=np.float64)
            if grades.ndim == 1:
                if grades.shape[0] != 2:
                    raise ValueError(
                        "multi_critical in-memory interface expects each filtration entry to be shape (2,) "
                        "or (k, 2)."
                    )
                grade_row_view = grades
                interface_input.filtration_values[i].push_back(
                    pair[double, double](grade_row_view[0], grade_row_view[1])
                )
            elif grades.ndim == 2:
                if grades.shape[1] != 2:
                    raise ValueError(
                        "multi_critical in-memory interface expects each filtration entry to be shape (2,) "
                        "or (k, 2)."
                    )
                grades_view_2d = grades
                interface_input.filtration_values[i].reserve(grades_view_2d.shape[0])
                for j in range(grades_view_2d.shape[0]):
                    interface_input.filtration_values[i].push_back(
                        pair[double, double](grades_view_2d[j, 0], grades_view_2d[j, 1])
                    )
            else:
                raise ValueError(
                    "multi_critical in-memory interface expects each filtration entry to be shape (2,) "
                    "or (k, 2)."
                )

    return interface_input


def one_criticalify(
    slicer,
    bint reduce=False,
    str algo="path",
    degree=None,
    swedish=None,
    bint verbose=False,
    bint kcritical=False,
    str filtration_container="contiguous",
    **slicer_kwargs,
):
    if not multi_critical_interface_available():
        raise RuntimeError("multi_critical in-memory interface is not available.")

    cdef bint use_logpath
    cdef bint use_swedish
    cdef multi_critical_interface_input_data interface_input
    cdef multi_critical_interface_output_data one_output
    cdef vector[multi_critical_interface_output_data] all_outputs
    cdef int target_degree
    cdef Py_ssize_t i
    cdef object out
    cdef intptr_t input_ptr
    cdef intptr_t out_ptr
    cdef C_KContiguousSlicer_Matrix0_f64* input_cpp
    cdef C_ContiguousSlicer_Matrix0_f64* out_cpp

    swedish = degree is not None if swedish is None else swedish
    from multipers import Slicer
    newSlicer = Slicer(
        slicer,
        return_type_only=True,
        kcritical=kcritical,
        filtration_container=filtration_container,
        **slicer_kwargs,
    )

    use_logpath = algo != "path"
    use_swedish = swedish is True

    if (not reduce) and isinstance(slicer, mps._KContiguousSlicer_Matrix0_f64):
        out = mps._ContiguousSlicer_Matrix0_f64()
        input_ptr = <intptr_t>(slicer.get_ptr())
        out_ptr = <intptr_t>(out.get_ptr())
        input_cpp = <C_KContiguousSlicer_Matrix0_f64*>input_ptr
        out_cpp = <C_ContiguousSlicer_Matrix0_f64*>out_ptr
        with nogil:
            out_cpp[0] = multi_critical_resolution_contiguous_interface_cpp(
                dereference(input_cpp),
                use_logpath,
                True,
                verbose,
            )
        if newSlicer is type(out):
            return out
        return newSlicer(out)

    interface_input = _multi_critical_input_from_slicer(slicer)
    if reduce and degree is None:
        with nogil:
            all_outputs = multi_critical_minpres_all_interface(
                interface_input,
                use_logpath,
                True,
                verbose,
                use_swedish,
            )
        return tuple(
            _slicer_from_multi_critical_output(newSlicer, all_outputs[i], True, i)
            for i in range(all_outputs.size())
        )
    elif reduce:
        _degree = <int>degree
        with nogil:
            one_output = multi_critical_minpres_interface(
                interface_input,
                _degree, 
                use_logpath,
                True,
                verbose,
                use_swedish,
            )
        return _slicer_from_multi_critical_output(newSlicer, one_output, True,degree)
    else:
        with nogil:
            one_output = multi_critical_resolution_interface(
                interface_input,
                use_logpath,
                True,
                verbose,
            )
        return _slicer_from_multi_critical_output(newSlicer, one_output)
