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

cdef extern from "multi_parameter_rank_invariant/euler_characteristic.h" namespace "Gudhi::multiparameter::euler_characteristic":
	void get_euler_surface_python(const intptr_t, tensor_dtype*, const vector[indices_type], bool, bool, bool) except + nogil
	signed_measure_type get_euler_signed_measure(const intptr_t, tensor_dtype* , const vector[indices_type], bool, bool) except + nogil

def euler_signed_measure(simplextree, mass_default=None, bool verbose=False, bool plot=False, grid_conversion=None):
	"""
	Computes the signed measures given by the decomposition of the hilbert function.

	Input
	-----
	 - simplextree:SimplexTreeMulti, the multifiltered simplicial complex
	 - mass_default: Either None, or 'auto' or 'inf', or array-like of floats. Where to put the default mass to get a zero-mass measure.
	 - plot:bool, plots the computed measures if true.
	 - n_jobs:int, number of jobs. Defaults to #cpu, but when doing parallel computations of signed measures, we recommend setting this to 1.
	 - verbose:bool, prints c++ logs.
	
	Output
	------
	`[signed_measure_of_degree for degree in degrees]`
	with `signed_measure_of_degree` of the form `(dirac location, dirac weights)`.
	"""
	assert len(simplextree.filtration_grid[0]) > 0, "Squeeze grid first."
	cdef bool zero_pad = mass_default is not None
	# assert simplextree.num_parameters == 2
	grid_shape = np.array([len(f) for f in simplextree.filtration_grid])
	
	# match mass_default: ## Cython bug
	# 	case None:
	# 		pass
	# 	case "inf":
	# 		mass_default = np.array([np.inf]*simplextree.num_parameters)
	# 	case "auto":
	# 		mass_default = np.array([1.1*np.max(f) - 0.1*np.min(f) for f in grid_conversion])
	# 	case _:
	# 		mass_default = np.asarray(mass_default)
	# 		assert mass_default.ndim == 1 and mass_default.shape[0] == simplextree.num_parameters
	if mass_default is None:
		mass_default = mass_default
	else:
		mass_default = np.asarray(mass_default)
		assert mass_default.ndim == 1 and mass_default.shape[0] == simplextree.num_parameters
	if zero_pad:
		for i, _ in enumerate(grid_shape):
			grid_shape[i] += 1 # adds a 0
		if grid_conversion is not None:
			for i,f in enumerate(grid_conversion):
				grid_conversion[i] = np.concatenate([f, [mass_default[i]]])
	assert len(grid_shape) == simplextree.num_parameters, "Grid shape size has to be the number of parameters."
	container_array = np.ascontiguousarray(np.zeros(grid_shape, dtype=python_tensor_dtype).flatten())
	assert len(container_array) < np.iinfo(python_indices_type).max, "Too large container. Raise an issue on github if you encounter this issue. (Due to tensor's operator[])"
	cdef intptr_t simplextree_ptr = simplextree.thisptr
	cdef vector[indices_type] c_grid_shape = grid_shape
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	cdef signed_measure_type out
	with nogil:
		out = get_euler_signed_measure(simplextree_ptr, container_ptr, c_grid_shape, zero_pad, verbose)
	pts, weights = np.asarray(out.first, dtype=int).reshape(-1, simplextree.num_parameters), np.asarray(out.second, dtype=int)
	# return pts, weights
	sm = (pts,weights)

	if grid_conversion is not None:
		from multipers.hilbert_function import sms_in_grid
		sm, = sms_in_grid([sm], grid_conversion)
	if plot:
		from multipers.plots import plot_signed_measures
		plot_signed_measures([sm])
	return sm


def euler_surface(simplextree, bool mobius_inversion=False, bool zero_pad=False, plot=False, bool verbose=False):
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
	assert len(simplextree.filtration_grid[0]) > 0, "Squeeze grid first."
	grid_conversion = [np.asarray(f) for f in simplextree.filtration_grid] if len(simplextree.filtration_grid[0]) > 0 else None
	# assert simplextree.num_parameters == 2
	grid_shape = [len(f) for f in grid_conversion]
	assert len(grid_shape) == simplextree.num_parameters
	container_array = np.ascontiguousarray(np.zeros(grid_shape, dtype=python_tensor_dtype).flatten())
	cdef intptr_t simplextree_ptr = simplextree.thisptr
	cdef vector[indices_type] c_grid_shape = grid_shape
	cdef tensor_dtype[::1] container = container_array
	cdef tensor_dtype* container_ptr = &container[0]
	# cdef signed_measure_type out
	# cdef indices_type i = 0
	# cdef indices_type j = 1
	# cdef vector[indices_type] fixed_values = np.asarray([0,0], dtype=int)
	with nogil:
		get_euler_surface_python(simplextree_ptr, container_ptr, c_grid_shape, mobius_inversion, zero_pad, verbose)
	out = (grid_conversion, container_array.reshape(grid_shape))
	if plot:
		from multipers.plots import plot_surface
		plot_surface(*out)
	return out

