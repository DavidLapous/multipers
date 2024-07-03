# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
from libcpp.utility cimport pair
from typing import Optional,Iterable,Callable
from multipers.grids import sms_in_grid
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

cdef extern from "multi_parameter_rank_invariant/hilbert_function.h" namespace "Gudhi::multiparameter::hilbert_function":
	void get_hilbert_surface_python(const intptr_t, tensor_dtype* , const vector[indices_type], const vector[indices_type], bool, bool, indices_type, bool) except + nogil
	signed_measure_type get_hilbert_signed_measure(const intptr_t, tensor_dtype* , const vector[indices_type], const vector[indices_type], bool, indices_type, bool, bool) except + nogil



def hilbert_signed_measure(
		simplextree,
		vector[indices_type] degrees, 
		mass_default=None, 
		plot=False, 
		indices_type n_jobs=0, 
		bool verbose=False,
		bool expand_collapse=False, 
		grid_conversion = None
	):
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
	assert simplextree.is_squeezed, "Squeeze grid first."
	cdef bool zero_pad = mass_default is not None
	# assert simplextree.num_parameters == 2
	grid_shape = np.array([len(f) for f in simplextree.filtration_grid])
	if mass_default is None:
		mass_default = mass_default
	else:
		mass_default = np.asarray(mass_default)
		assert mass_default.ndim == 1 and mass_default.shape[0] == simplextree.num_parameters
	if zero_pad:
		for i, _ in enumerate(grid_shape):
			grid_shape[i] += 1 # adds a 0
		if  grid_conversion is not None:
			for i,f in enumerate(grid_conversion):
				grid_conversion[i] = np.concatenate([f, [mass_default[i]]])
	assert len(grid_shape) == simplextree.num_parameters, "Grid shape size has to be the number of parameters."
	grid_shape_with_degree = np.asarray(np.concatenate([[len(degrees)], grid_shape]), dtype=python_indices_type)
	container_array = np.ascontiguousarray(np.zeros(grid_shape_with_degree, dtype=python_tensor_dtype).flatten())
	assert len(container_array) < np.iinfo(np.uint32).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
	cdef intptr_t simplextree_ptr = simplextree.thisptr
	cdef vector[indices_type] c_grid_shape = grid_shape_with_degree
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	cdef signed_measure_type out
	with nogil:
		out = get_hilbert_signed_measure(simplextree_ptr, container_ptr, c_grid_shape, degrees, zero_pad, n_jobs, verbose, expand_collapse)
	pts, weights = np.asarray(out.first, dtype=int).reshape(-1, simplextree.num_parameters+1), np.asarray(out.second, dtype=int)
	# return pts, weights

	# degree_indices = [np.argwhere(pts[:,0] == degree_index).flatten() for degree_index, degree in enumerate(degrees)] ## TODO : maybe optimize
	# sms = [(pts[idx,1:], weights[idx]) for idx in degree_indices]
	
	# is_sorted = lambda a: np.all(a[:-1] <= a[1:])
	# assert is_sorted(pts[:,0]), "TODO : REMOVE THIS."
	slices = np.concatenate([np.searchsorted(pts[:,0], np.arange(degrees.size())), [pts.shape[0]] ])
	sms = [
			(pts[slices[i]:slices[i+1],1:],weights[slices[i]:slices[i+1]])
			for i in range(slices.shape[0]-1)
	]
	if grid_conversion is not None:
		sms = sms_in_grid(sms,grid_conversion)

	if plot:
		from multipers.plots import plot_signed_measures
		plot_signed_measures(sms)
	return sms



def hilbert_surface(simplextree, vector[indices_type] degrees, mass_default=None, bool mobius_inversion=False, bool plot=False, indices_type n_jobs=0, bool expand_collapse=False):
	"""
	Computes the hilbert function.

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
	Integer array of the form `(num_degrees, num_filtration_values_of_parameter 1, ..., num_filtration_values_of_parameter n)`
	"""
	assert simplextree.is_squeezed > 0, "Squeeze grid first."
	cdef bool zero_pad = mass_default is not None
	grid_conversion = [np.asarray(f) for f in simplextree.filtration_grid]
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
	grid_shape_with_degree = np.asarray(np.concatenate([[len(degrees)], grid_shape]), dtype=python_indices_type)
	container_array = np.ascontiguousarray(np.zeros(grid_shape_with_degree, dtype=python_tensor_dtype).flatten())
	assert len(container_array) < np.iinfo(np.uint32).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
	cdef intptr_t simplextree_ptr = simplextree.thisptr
	cdef vector[indices_type] c_grid_shape = grid_shape_with_degree
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	with nogil:
		get_hilbert_surface_python(simplextree_ptr, container_ptr, c_grid_shape, degrees, mobius_inversion, zero_pad, n_jobs, expand_collapse)
	out = (grid_conversion, container_array.reshape(grid_shape_with_degree))
	if plot:
		from multipers.plots import plot_surfaces
		plot_surfaces(out)
	return out


