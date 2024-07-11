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

from multipers.ml.signed_betti import rank_decomposition_by_rectangles
from multipers.point_measure_integration import sparsify



## TODO : It is not necessary to do the Möbius inversion in python.
## fill rank in flipped death, then differentiate in cpp, then reflip with numpy.
def rank_from_slicer(
        slicer, 
        vector[indices_type] degrees,
        bool verbose=False,
        indices_type n_jobs=1,
        bool zero_pad = False,
        grid_shape=None,
        bool plot=False,
        bool return_raw=False,
        ):
    # cdef intptr_t slicer_ptr = <intptr_t>(slicer.get_ptr())
    if grid_shape is None:
        grid_shape = (slicer.compute_box()[1]).astype(python_indices_type)
    grid_shape = np.asarray(grid_shape)+1

    cdef int num_parameters = len(grid_shape)

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
    rank = tuple(rank_decomposition_by_rectangles(rank_of_degree, threshold = zero_pad) for rank_of_degree in rank)
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






