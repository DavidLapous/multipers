import cython
from libc.stdint cimport intptr_t
from libcpp cimport bool
from cython.operator cimport dereference

import numpy as np
import multipers.slicer as mps
from multipers.slicer cimport C_ContiguousSlicer_Matrix0_f64

cdef extern from "ext_interface/contiguous_slicer_bridge.hpp" namespace "multipers":
  cdef cppclass contiguous_f64_complex_cpp "multipers::contiguous_f64_complex":
    contiguous_f64_complex_cpp() except + nogil
    contiguous_f64_complex_cpp(const contiguous_f64_complex_cpp&) except + nogil

  void assign_slicer_from_contiguous_f64_complex_cpp "multipers::assign_slicer_from_contiguous_f64_complex"(
      C_ContiguousSlicer_Matrix0_f64&,
      contiguous_f64_complex_cpp&
  ) except + nogil



cdef extern from "ext_interface/mpfree_interface.hpp" namespace "multipers":

  bool mpfree_interface_available "multipers::mpfree_interface_available"() except + nogil

  contiguous_f64_complex_cpp mpfree_minpres_contiguous_interface_cpp "multipers::mpfree_minpres_contiguous_interface"(
      C_ContiguousSlicer_Matrix0_f64&,
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
    cdef object input_slicer
    cdef object out
    cdef intptr_t input_ptr
    cdef intptr_t out_ptr
    cdef C_ContiguousSlicer_Matrix0_f64* input_cpp
    cdef C_ContiguousSlicer_Matrix0_f64* out_cpp
    cdef contiguous_f64_complex_cpp out_complex

    if not mpfree_interface_available():
        raise RuntimeError("mpfree in-memory interface is not available.")

    input_slicer = slicer
    if not isinstance(slicer, mps._ContiguousSlicer_Matrix0_f64):
        input_slicer = slicer.astype(
            vineyard=False,
            kcritical=False,
            dtype=np.float64,
            col=slicer.col_type,
            pers_backend="matrix",
            filtration_container="contiguous",
        )

    out = mps._ContiguousSlicer_Matrix0_f64()
    input_ptr = <intptr_t>(input_slicer.get_ptr())
    out_ptr = <intptr_t>(out.get_ptr())
    input_cpp = <C_ContiguousSlicer_Matrix0_f64*>input_ptr
    out_cpp = <C_ContiguousSlicer_Matrix0_f64*>out_ptr

    with nogil:
        out_complex = mpfree_minpres_contiguous_interface_cpp(
            dereference(input_cpp),
            degree,
            full_resolution,
            use_chunk,
            use_clearing,
            verbose,
        )
        assign_slicer_from_contiguous_f64_complex_cpp(out_cpp[0], out_complex)

    if isinstance(slicer, mps._ContiguousSlicer_Matrix0_f64):
        return out
    return out.astype(
        vineyard=slicer.is_vine,
        kcritical=slicer.is_kcritical,
        dtype=slicer.dtype,
        col=slicer.col_type,
        pers_backend=slicer.pers_backend,
        filtration_container=slicer.filtration_container,
    )
