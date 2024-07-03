# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t, int16_t, int8_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
from libcpp.utility cimport pair
from typing import Optional,Iterable,Callable
from cython.operator import dereference

import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef float value_type 
python_value_type=np.float32

ctypedef int32_t indices_type # uint fails for some reason
python_indices_type=np.int32

ctypedef int32_t tensor_dtype # sizes should be less than 32k (int16), but int32 to be safe
python_tensor_dtype = np.int32
import multipers.grids as mpg
import multipers.slicer as mps
from multipers.slicer cimport *

# ctypedef pair[vector[vector[indices_type]], vector[tensor_dtype]] signed_measure_type

# cdef extern from "multi_parameter_rank_invariant/rank_invariant.h" namespace "Gudhi::multiparameter::rank_invariant":
    # void compute_rank_invariant_python(const intptr_t, tensor_dtype* , const vector[indices_type], const vector[indices_type], indices_type, bool) except + nogil

    # void compute_rank_invariant_python(GeneralNoVineTruc, tensor_dtype* , const vector[indices_type], const vector[indices_type], indices_type) except + nogil
    # void compute_rank_invariant_python(GeneralVineTruc, tensor_dtype* , const vector[indices_type], const vector[indices_type], indices_type) except + nogil
    # void compute_rank_invariant_python(SimplicialVineMatrixTruc, tensor_dtype* , const vector[indices_type], const vector[indices_type], indices_type) except + nogil
    # void compute_rank_invariant_python(SimplicialVineGraphTruc, tensor_dtype* , const vector[indices_type], const vector[indices_type], indices_type) except + nogil
    # void compute_rank_invariant_python(GeneralVineClementTruc, tensor_dtype* , const vector[indices_type], const vector[indices_type], indices_type) except + nogil

from multipers.ml.signed_betti import rank_decomposition_by_rectangles
from multipers.point_measure_integration import sparsify

# def rank_invariant(simplextree, vector[indices_type] degrees, mass_default=None, plot=False, indices_type n_jobs=0, bool verbose=False, bool expand_collapse=False):
#     """
#     Computes the signed measures given by the decomposition of the hilbert function.
#
#     Input
#     -----
#      - simplextree:SimplexTreeMulti, the multifiltered simplicial complex
#      - degrees:array-like of ints, the degrees to compute
#      - mass_default: Either None, or 'auto' or 'inf', or array-like of floats. Where to put the default mass to get a zero-mass measure.
#      - plot:bool, plots the computed measures if true.
#      - n_jobs:int, number of jobs. Defaults to #cpu, but when doing parallel computations of signed measures, we recommend setting this to 1.
#      - verbose:bool, prints c++ logs.
#
#     Output
#     ------
#     `[signed_measure_of_degree for degree in degrees]`
#     with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.
#     """
#     assert simplextree.is_squeezed, "Squeeze grid first."
#     assert simplextree.dtype == np.int32
#     cdef bool zero_pad = mass_default is not None
#     grid_conversion = [np.asarray(f) for f in simplextree.filtration_grid]
#     # assert simplextree.num_parameters == 2
#     grid_shape = np.array([len(f) for f in grid_conversion])
#
#     if mass_default is None:
#         mass_default = mass_default
#     else:
#         mass_default = np.asarray(mass_default)
#         assert mass_default.ndim == 1 and mass_default.shape[0] == simplextree.num_parameters
#     if zero_pad:
#         for i, _ in enumerate(grid_shape):
#             grid_shape[i] += 1 # adds a 0
#         for i,f in enumerate(grid_conversion):
#             grid_conversion[i] = np.concatenate([f, [mass_default[i]]])
#
#     assert len(grid_shape) == simplextree.num_parameters, "Grid shape size has to be the number of parameters."
#     grid_shape_with_degree = np.asarray(np.concatenate([[len(degrees)], grid_shape, grid_shape]), dtype=python_indices_type)
#     container_array = np.ascontiguousarray(np.zeros(grid_shape_with_degree, dtype=python_tensor_dtype).flatten())
#     assert len(container_array) < np.iinfo(python_indices_type).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
#     cdef intptr_t simplextree_ptr = simplextree.thisptr
#     cdef vector[indices_type] c_grid_shape = grid_shape_with_degree
#     cdef tensor_dtype[::1] container = container_array
#     cdef tensor_dtype* container_ptr = &container[0]
#     with nogil:
#         compute_rank_invariant_python(simplextree_ptr, container_ptr,c_grid_shape,degrees, n_jobs, expand_collapse)
#     container_array = container_array.reshape(grid_shape_with_degree)
#     if plot:
#         from multipers.plots import plot_surfaces
#         plot_surfaces((grid_conversion, container_array))
#     return (grid_conversion, container_array)
#
#
# def signed_measure(simplextree, vector[indices_type] degrees, mass_default=None, plot=False, indices_type n_jobs=0, bool verbose=False, bool expand_collapse=False):
#     """
#     Computes the signed measures given by the decomposition of the hilbert function.
#
#     Input
#     -----
#      - simplextree:SimplexTreeMulti, the multifiltered simplicial complex
#      - degrees:array-like of ints, the degrees to compute
#      - mass_default: Either None, or 'auto' or 'inf', or array-like of floats. Where to put the default mass to get a zero-mass measure.
#      - plot:bool, plots the computed measures if true.
#      - n_jobs:int, number of jobs. Defaults to #cpu, but when doing parallel computations of signed measures, we recommend setting this to 1.
#      - verbose:bool, prints c++ logs.
#
#     Output
#     ------
#     `[signed_measure_of_degree for degree in degrees]`
#     with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.
#     """
#     assert simplextree.is_squeezed, "Squeeze grid first."
#     cdef bool zero_pad = mass_default is not None
#     grid_conversion = [np.asarray(f) for f in simplextree.filtration_grid]
#     # assert simplextree.num_parameters == 2
#     grid_shape = np.array([len(f) for f in grid_conversion])
#
#     if mass_default is None:
#         mass_default = mass_default
#     else:
#         mass_default = np.asarray(mass_default)
#         assert mass_default.ndim == 1 and mass_default.shape[0] == simplextree.num_parameters, "Mass default has to be an array like of shape (num_parameters,)"
#     if zero_pad:
#         for i, _ in enumerate(grid_shape):
#             grid_shape[i] += 1 # adds a 0
#         for i,f in enumerate(grid_conversion):
#             grid_conversion[i] = np.concatenate([f, [mass_default[i]]])
#
#     assert len(grid_shape) == simplextree.num_parameters, "Grid shape size has to be the number of parameters."
#     grid_shape_with_degree = np.asarray(np.concatenate([[len(degrees)], grid_shape, grid_shape]), dtype=python_indices_type)
#     container_array = np.ascontiguousarray(np.zeros(grid_shape_with_degree, dtype=python_tensor_dtype).flatten())
#     assert len(container_array) < np.iinfo(python_indices_type).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
#     cdef intptr_t simplextree_ptr = simplextree.thisptr
#     cdef vector[indices_type] c_grid_shape = grid_shape_with_degree
#     cdef tensor_dtype[::1] container = container_array
#     cdef tensor_dtype* container_ptr = &container[0]
#     with nogil:
#         compute_rank_invariant_python(simplextree_ptr, container_ptr,c_grid_shape,degrees, n_jobs, expand_collapse)
#     rank = container_array.reshape(grid_shape_with_degree)
#     rank = tuple(rank_decomposition_by_rectangles(rank_of_degree) for rank_of_degree in rank)
#     out = []
#     cdef int num_parameters = simplextree.num_parameters
#     for rank_decomposition in rank:
#         (coords, weights) = sparsify(np.ascontiguousarray(rank_decomposition))
#         births = coords[:,:num_parameters]
#         deaths = coords[:,num_parameters:]
#         correct_indices = np.all(births<=deaths, axis=1) # TODO : correct this
#         coords = coords[correct_indices]
#         weights = weights[correct_indices]
#         if len(correct_indices) == 0:
#             pts, weights = np.empty((0, 2*num_parameters)), np.empty((0))
#         else:
#             pts = np.empty(shape=coords.shape, dtype=grid_conversion[0].dtype)
#             for i in range(pts.shape[1]):
#                 pts[:,i] = grid_conversion[i % num_parameters][coords[:,i]]
#         rank_decomposition = (pts,weights)
#         out.append(rank_decomposition)
#
#     if plot:
#         from multipers.plots import plot_signed_measures
#         plot_signed_measures(out)
#     return out



## TODO : It is not necessary to do the Möbius inversion in python.
## fill rank in flipped death, then differentiate in cpp, then reflip with numpy.
def rank_from_slicer(
        slicer, 
        vector[indices_type] degrees,
        bool verbose=False,
        indices_type n_jobs=1,
        mass_default = None,
        grid_shape=None,
        bool plot=False,
        bool return_raw=False,
        ):
    # cdef intptr_t slicer_ptr = <intptr_t>(slicer.get_ptr())
    if grid_shape is None:
        grid_shape = (slicer.compute_box()[1]).astype(python_indices_type)
    cdef int num_parameters = len(grid_shape)
    cdef bool zero_pad = mass_default is not None

    if mass_default is None:
        mass_default = mass_default
    else:
        mass_default = np.asarray(mass_default)
        assert mass_default.ndim == 1 and mass_default.shape[0] == num_parameters, "Mass default has to be an array like of shape (num_parameters,)"
    if zero_pad:
        for i, _ in enumerate(grid_shape):
            grid_shape[i] += 1 # adds a 0
        # for i,f in enumerate(grid_conversion):
        #     grid_conversion[i] = np.concatenate([f, [mass_default[i]]])

    grid_shape_with_degree = np.asarray(np.concatenate([[len(degrees)], grid_shape, grid_shape]), dtype=python_indices_type)
    container_array = np.ascontiguousarray(np.zeros(grid_shape_with_degree, dtype=python_tensor_dtype).flatten())
    assert len(container_array) < np.iinfo(python_indices_type).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
    cdef vector[indices_type] c_grid_shape = grid_shape_with_degree
    cdef tensor_dtype[::1] container = container_array
    cdef tensor_dtype* container_ptr = &container[0]

    ## SLICERS
    _compute_rank_invariant(slicer, container_ptr, c_grid_shape, degrees, n_jobs)

    rank = container_array.reshape(grid_shape_with_degree)
    rank = tuple(rank_decomposition_by_rectangles(rank_of_degree) for rank_of_degree in rank)
    if return_raw:
        return rank
    out = []
    def clean_rank(rank_decomposition):
        (coords, weights) = sparsify(np.ascontiguousarray(rank_decomposition))
        births = coords[:,:num_parameters]
        deaths = coords[:,num_parameters:]
        correct_indices = np.all(births<=deaths, axis=1)
        coords = coords[correct_indices]
        weights = weights[correct_indices]
        return coords, weights

    out = tuple(clean_rank(rank_decomposition) for rank_decomposition in rank)
    return out






