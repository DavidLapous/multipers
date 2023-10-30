# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
from libcpp.utility cimport pair
from typing import Optional,Iterable,Callable


ctypedef float value_type 
# ctypedef uint16_t index_type 

import numpy as np
# cimport numpy as cnp
# cnp.import_array()

# cdef extern from "multi_parameter_rank_invariant/rank_invariant.h" namespace "Gudhi::rank_invariant":
#     void get_hilbert_surface(const intptr_t, mt.static_tensor_view, const vector[index_type], const vector[index_type], index_type, index_type, const vector[index_type], bool, bool) except + nogil


from multipers.simplex_tree_multi import SimplexTreeMulti


def numpy_to_tensor(array:np.ndarray):
    cdef vector[index_type] shape = array.shape
    cdef dtype[::1] contigus_array_view = np.ascontiguousarray(array)
    cdef dtype* dtype_ptr = &contigus_array_view[0]
    cdef mt.static_tensor_view tensor
    with nogil:
        tensor = mt.static_tensor_view(dtype_ptr, shape)
    return tensor.get_resolution()

# def hilbert2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, vector[index_type] degrees, bool mobius_inversion):
# 	# assert simplextree.num_parameters == 2
# 	cdef intptr_t ptr = simplextree.thisptr
# 	cdef vector[index_type] c_grid_shape = grid_shape
# 	cdef dtype[::1] container = np.zeros(grid_shape, dtype=np.float32).flatten()
# 	cdef dtype* container_ptr = &container[0]
# 	cdef mt.static_tensor_view c_container = mt.static_tensor_view(container_ptr, c_grid_shape)
# 	cdef index_type i = 0
# 	cdef index_type j = 1
# 	cdef vector[index_type] fixed_values = [[],[]]
# 	# get_hilbert_surface(ptr, c_container, c_grid_shape, degrees,i,j,fixed_values, False, False)
# 	return container.reshape(grid_shape)

