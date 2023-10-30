# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
from libcpp.utility cimport pair
from typing import Optional,Iterable,Callable

def hilbert_signed_measure(simplextree, degrees, mass_default=None, plot=False, n_jobs=0, verbose=False):
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
	pass


def hilbert_function(simplextree, degrees, zero_pad=False, plot=False, n_jobs=0):
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
	pass