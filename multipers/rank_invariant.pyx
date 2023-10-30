# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
from libcpp.utility cimport pair
from typing import Optional,Iterable,Callable

import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef float value_type 
python_value_type=np.float32

ctypedef int32_t indices_type # uint fails for some reason
python_indices_type=np.int32

ctypedef int32_t tensor_dtype
python_tensor_dtype = np.int32


ctypedef pair[vector[vector[indices_type]], vector[tensor_dtype]] signed_measure_type

cdef extern from "multi_parameter_rank_invariant/rank_invariant.h" namespace "Gudhi::multiparameter::rank_invariant":
	void compute_rank_invariant_python(const intptr_t, tensor_dtype* , const vector[indices_type], const vector[indices_type], indices_type, bool) except + nogil
	

def rank_invariant(simplextree, vector[indices_type] degrees, mass_default=None, plot=False, indices_type n_jobs=0, bool verbose=False, bool expand_collapse=False):
	"""
	Computes the signed measures given by the decomposition of the hilbert function.

	Input
	-----
	 - simplextree:SimplexTreeMulti, the multifiltered simplicial complex
	 - degrees:array-like of ints, the degrees to compute
	 - mass_default: Either None, or 'auto' or 'inf', or array-like of floats. Where to put the default mass to get a zero-mass measure.
	 - plot:bool, plots the computed measures if true.
	 - n_jobs:int, number of jobs. Defaults to #cpu, but when doing parallel computations of signed measures, we recommend setting this to 1.
	 - verbose:bool, prints c++ logs.
	
	Output
	------
	`[signed_measure_of_degree for degree in degrees]`
	with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.
	"""
	assert simplextree._is_squeezed, "Squeeze grid first."
	cdef bool zero_pad = mass_default is not None
	grid_conversion = [np.asarray(f) for f in simplextree.filtration_grid]
	# assert simplextree.num_parameters == 2
	grid_shape = np.array([len(f) for f in grid_conversion])
	
	if mass_default is None:
		mass_default = mass_default
	else:
		mass_default = np.asarray(mass_default)
		assert mass_default.ndim == 1 and mass_default.shape[0] == simplextree.num_parameters
	if zero_pad:
		for i, _ in enumerate(grid_shape):
			grid_shape[i] += 1 # adds a 0
		for i,f in enumerate(grid_conversion):
			grid_conversion[i] = np.concatenate([f, [mass_default[i]]])
		
	assert len(grid_shape) == simplextree.num_parameters, "Grid shape size has to be the number of parameters."
	grid_shape_with_degree = np.asarray(np.concatenate([[len(degrees)], grid_shape, grid_shape]), dtype=python_indices_type)
	container_array = np.ascontiguousarray(np.zeros(grid_shape_with_degree, dtype=python_tensor_dtype).flatten())
	assert len(container_array) < np.iinfo(python_indices_type).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
	cdef intptr_t simplextree_ptr = simplextree.thisptr
	cdef vector[indices_type] c_grid_shape = grid_shape_with_degree
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	with nogil:
		compute_rank_invariant_python(simplextree_ptr, container_ptr,c_grid_shape,degrees, n_jobs, expand_collapse)
	container_array = container_array.reshape(grid_shape_with_degree)
	if plot:
		from multipers.plots import plot_surfaces
		plot_surfaces((grid_conversion, container_array))
	return (grid_conversion, container_array)


def signed_measure(simplextree, vector[indices_type] degrees, mass_default=None, plot=False, indices_type n_jobs=0, bool verbose=False, bool expand_collapse=False):
	"""
	Computes the signed measures given by the decomposition of the hilbert function.

	Input
	-----
	 - simplextree:SimplexTreeMulti, the multifiltered simplicial complex
	 - degrees:array-like of ints, the degrees to compute
	 - mass_default: Either None, or 'auto' or 'inf', or array-like of floats. Where to put the default mass to get a zero-mass measure.
	 - plot:bool, plots the computed measures if true.
	 - n_jobs:int, number of jobs. Defaults to #cpu, but when doing parallel computations of signed measures, we recommend setting this to 1.
	 - verbose:bool, prints c++ logs.
	
	Output
	------
	`[signed_measure_of_degree for degree in degrees]`
	with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.
	"""
	assert simplextree._is_squeezed > 0, "Squeeze grid first."
	cdef bool zero_pad = mass_default is not None
	grid_conversion = [np.asarray(f) for f in simplextree.filtration_grid]
	# assert simplextree.num_parameters == 2
	grid_shape = np.array([len(f) for f in grid_conversion])
	
	if mass_default is None:
		mass_default = mass_default
	else:
		mass_default = np.asarray(mass_default)
		assert mass_default.ndim == 1 and mass_default.shape[0] == simplextree.num_parameters, "Mass default has to be an array like of shape (num_parameters,)"
	if zero_pad:
		for i, _ in enumerate(grid_shape):
			grid_shape[i] += 1 # adds a 0
		for i,f in enumerate(grid_conversion):
			grid_conversion[i] = np.concatenate([f, [mass_default[i]]])
		
	assert len(grid_shape) == simplextree.num_parameters, "Grid shape size has to be the number of parameters."
	grid_shape_with_degree = np.asarray(np.concatenate([[len(degrees)], grid_shape, grid_shape]), dtype=python_indices_type)
	container_array = np.ascontiguousarray(np.zeros(grid_shape_with_degree, dtype=python_tensor_dtype).flatten())
	assert len(container_array) < np.iinfo(python_indices_type).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
	cdef intptr_t simplextree_ptr = simplextree.thisptr
	cdef vector[indices_type] c_grid_shape = grid_shape_with_degree
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	with nogil:
		compute_rank_invariant_python(simplextree_ptr, container_ptr,c_grid_shape,degrees, n_jobs, expand_collapse)
	rank = container_array.reshape(grid_shape_with_degree)
	from multipers.ml.signed_betti import rank_decomposition_by_rectangles
	from torch import Tensor
	rank = [rank_decomposition_by_rectangles(rank_of_degree) for rank_of_degree in rank]
	out = []
	cdef int num_parameters = simplextree.num_parameters
	for rank_decomposition in rank:
		rank_decomposition = Tensor(np.ascontiguousarray(rank_decomposition)).to_sparse()
		def _is_trivial(rectangle:np.ndarray):
			birth=rectangle[:num_parameters]
			death=rectangle[num_parameters:]
			return np.all(birth<=death) and not np.array_equal(birth,death)
		coords = np.asarray(rank_decomposition.indices().T, dtype=int)
		weights = np.asarray(rank_decomposition.values(), dtype=int)
		correct_indices = np.asarray([_is_trivial(rectangle) for rectangle in coords])
		coords = coords[correct_indices]
		weights = weights[correct_indices]
		if len(correct_indices) == 0:
			pts, weights = np.empty((0, 2*num_parameters)), np.empty((0))
		else:
			pts = np.empty(shape=coords.shape, dtype=grid_conversion[0].dtype)
			for i in range(pts.shape[1]):
				pts[:,i] = grid_conversion[i % num_parameters][coords[:,i]]
		rank_decomposition = (pts,weights)
		out.append(rank_decomposition)
	
	if plot:
		from multipers.plots import plot_signed_measures
		plot_signed_measures(out)
	return out