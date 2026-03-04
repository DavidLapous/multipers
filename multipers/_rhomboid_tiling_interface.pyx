import cython
import numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport intptr_t
from libc.string cimport memcpy
from cython.operator cimport dereference

import multipers.slicer as mps
from multipers.slicer cimport C_ContiguousSlicer_Matrix0_f64


cdef extern from "ext_interface/rhomboid_tiling_interface.hpp" namespace "multipers":

  cdef cppclass rhomboid_tiling_interface_input_data "multipers::rhomboid_tiling_interface_input<int>":
    rhomboid_tiling_interface_input_data() except + nogil
    vector[vector[double]] points

  bool rhomboid_tiling_interface_available "multipers::rhomboid_tiling_interface_available"() except + nogil

  C_ContiguousSlicer_Matrix0_f64 rhomboid_tiling_to_contiguous_slicer_interface_cpp "multipers::rhomboid_tiling_to_contiguous_slicer_interface<int>"(
      const rhomboid_tiling_interface_input_data&,
      int,
      int,
      bool
  ) except + nogil


def _is_available():
    return rhomboid_tiling_interface_available()


@cython.boundscheck(False)
@cython.wraparound(False)
def rhomboid_tiling_to_slicer(
    slicer,
    object point_cloud,
    int k_max,
    int degree,
    bint verbose=False,
):
    cdef object target
    cdef intptr_t target_ptr
    cdef C_ContiguousSlicer_Matrix0_f64* target_cpp

    cdef object point_cloud_array = np.ascontiguousarray(point_cloud, dtype=np.double)
    if point_cloud_array.ndim != 2:
        raise ValueError(
            f"point_cloud should be a 2d array. Got {point_cloud_array.shape=}"
        )

    cdef double[:, ::1] point_cloud_view = point_cloud_array
    cdef Py_ssize_t num_pts = point_cloud_view.shape[0]
    cdef Py_ssize_t dim = point_cloud_view.shape[1]
    cdef Py_ssize_t i

    cdef rhomboid_tiling_interface_input_data interface_input

    interface_input.points.resize(num_pts)
    for i in range(num_pts):
        interface_input.points[i].resize(dim)
        if dim > 0:
            memcpy(&interface_input.points[i][0], &point_cloud_view[i, 0], dim * sizeof(double))

    target = slicer
    if not isinstance(slicer, mps._ContiguousSlicer_Matrix0_f64):
        target = mps._ContiguousSlicer_Matrix0_f64()

    target_ptr = <intptr_t>(target.get_ptr())
    target_cpp = <C_ContiguousSlicer_Matrix0_f64*>target_ptr

    with nogil:
        target_cpp[0] = rhomboid_tiling_to_contiguous_slicer_interface_cpp(
            interface_input,
            k_max,
            degree,
            verbose,
        )

    if target is not slicer:
        slicer._from_ptr(type(slicer)(target).get_ptr())
    return slicer
