# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
from libcpp.utility cimport pair, tuple
from typing import Optional,Iterable,Callable

import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef double value_type 
python_value_type=np.float64

ctypedef int32_t indices_type # uint fails for some reason
python_indices_type=np.int32

ctypedef int32_t tensor_dtype
python_tensor_dtype = np.int32

ctypedef pair[vector[vector[indices_type]], vector[tensor_dtype]] signed_measure_type


from multipers.simplex_tree_multi import SimplexTreeMulti_Ff64
from gudhi.simplex_tree import SimplexTree

cdef extern from "multi_parameter_rank_invariant/function_rips.h" namespace "Gudhi::multiparameter::function_rips":
	void compute_function_rips_surface_python(const intptr_t, tensor_dtype* , const vector[indices_type], indices_type,indices_type, bool, bool, indices_type) except + nogil
	signed_measure_type compute_function_rips_signed_measure_python(const intptr_t, tensor_dtype* , const vector[indices_type], indices_type,indices_type, bool, bool, indices_type) except + nogil
	pair[vector[value_type],int] get_degree_rips_st_python(const intptr_t,const intptr_t, const vector[int]) except + nogil


import multipers.grids as mpg



def get_degree_rips(st, vector[int] degrees, grid_strategy="exact", resolution=0):
	assert isinstance(st,SimplexTree), "Input has to be a Gudhi simplextree for now."
	assert st.dimension() == 1, "Simplextree has to be of dimension 1. You can use the `prune_above_dimension` method."
	degree_rips_st = SimplexTreeMulti_Ff64(num_parameters=degrees.size())
	cdef intptr_t simplextree_ptr = st.thisptr
	cdef intptr_t st_multi_ptr = degree_rips_st.thisptr
	cdef pair[vector[value_type],int] out
	with nogil:
		out = get_degree_rips_st_python(simplextree_ptr, st_multi_ptr, degrees)
	filtrations = np.asarray(out.first)
	cdef int max_degree = out.second
	cdef bool inf_flag = filtrations[-1] == np.inf
	if inf_flag:
		filtrations = filtrations[:-1]
	filtrations, = mpg.compute_grid([filtrations],strategy=grid_strategy,resolution=resolution)
	if inf_flag:
		filtrations = np.concatenate([filtrations, [np.inf]])
	degree_rips_st.grid_squeeze([filtrations]*degree_rips_st.num_parameters, inplace=True, coordinate_values=True)
	degree_rips_st.filtration_grid = [filtrations, np.asarray(degrees)[::-1]]
	degree_rips_st._is_function_simplextree=True
	return degree_rips_st,max_degree

def function_rips_surface(st_multi, vector[indices_type] homological_degrees, bool mobius_inversion=True, bool zero_pad=False, indices_type n_jobs=0):
	assert st_multi.is_squeezed, "Squeeze first !"
	cdef intptr_t st_multi_ptr = st_multi.thisptr
	cdef indices_type I = len(st_multi.filtration_grid[0])
	cdef indices_type J = st_multi.num_parameters
	container_shape = (homological_degrees.size(),I,J)
	container_array = np.ascontiguousarray(np.zeros(container_shape, dtype=python_tensor_dtype).flatten())
	assert len(container_array) < np.iinfo(np.uint32).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	with nogil:
		compute_function_rips_surface_python(st_multi_ptr,container_ptr, homological_degrees, I,J, mobius_inversion, zero_pad, n_jobs)
	filtration_grid = st_multi.filtration_grid
	if filtration_grid[0][-1] == np.inf:
		filtration_grid[0][-1] = filtration_grid[0][-2]
	return filtration_grid, container_array.reshape(container_shape)



def function_rips_signed_measure(st_multi, vector[indices_type] homological_degrees, bool mobius_inversion=True, bool zero_pad=False, indices_type n_jobs=0, bool reconvert = True):
	assert st_multi.is_squeezed
	cdef intptr_t st_multi_ptr = st_multi.thisptr
	cdef indices_type I = len(st_multi.filtration_grid[0])
	cdef indices_type J = st_multi.num_parameters
	container_shape = (homological_degrees.size(),I,J)
	container_array = np.ascontiguousarray(np.zeros(container_shape, dtype=python_tensor_dtype).flatten())
	assert len(container_array) < np.iinfo(np.uint32).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	cdef signed_measure_type out 
	# TODO nogil
	with nogil:
		out = compute_function_rips_signed_measure_python(st_multi_ptr,container_ptr, homological_degrees, I,J, mobius_inversion, zero_pad, n_jobs)
	pts, weights = np.asarray(out.first, dtype=int).reshape(-1, 3), np.asarray(out.second, dtype=int)

	degree_indices = [np.argwhere(pts[:,0] == degree_index).flatten() for degree_index, degree in enumerate(homological_degrees)] ## TODO : maybe optimize
	sms = [(pts[id,1:],weights[id]) for id in degree_indices]
	if not reconvert: return sms

	grid_conversion = st_multi.filtration_grid
	for degree_index,(pts,weights) in enumerate(sms):
		coords = np.empty(shape=pts.shape, dtype=float)
		for i in range(coords.shape[1]):
			coords[:,i] = np.asarray(grid_conversion[i])[pts[:,i]]
		sms[degree_index]=(coords, weights)
	
	return sms
