"""!
@package mma
@brief Files containing the C++ cythonized functions.
@author David Loiseaux
@copyright Copyright (c) 2022 Inria.
"""

# distutils: language = c++

###########################################################################
## PYTHON LIBRARIES
import gudhi as gd
import numpy as np
from typing import List
import pickle as pk

###########################################################################
## CPP CLASSES
from libc.stdint cimport intptr_t
from libc.stdint cimport uintptr_t

###########################################################################
## CYTHON TYPES
from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from typing import Iterable
from cython.operator import dereference
#########################################################################
## Multipersistence Module Approximation Classes
from multipers.mma_structures cimport Module, Box, pair, boundary_matrix
from multipers.slicer cimport *


#########################################################################
## Small hack for typing
from gudhi import SimplexTree
from multipers.simplex_tree_multi import SimplexTreeMulti
from multipers.slicer import Slicer
from multipers.mma_structures import PyModule
# cimport numpy as cnp
# cnp.import_array()

###################################### MMA
cdef extern from "multiparameter_module_approximation/approximation.h" namespace "Gudhi::multiparameter::mma":
	Module compute_vineyard_barcode_approximation(boundary_matrix, vector[Finitely_critical_multi_filtration] , value_type precision, Box[value_type] &, bool threshold, bool complete, bool multithread, bool verbose) nogil
	# TODO : tempita
	Module multiparameter_module_approximation(SimplicialVineGraphTruc&, value_type, Box[value_type]&, bool, bool, bool) nogil

	Module multiparameter_module_approximation(SimplicialVineMatrixTruc&, value_type, Box[value_type]&, bool, bool, bool) nogil
	Module multiparameter_module_approximation(GeneralVineTruc&, value_type, Box[value_type]&, bool, bool, bool) nogil

# TODO : remove when old is deprecated
cdef extern from "multiparameter_module_approximation/format_python-cpp.h" namespace "Gudhi::multiparameter::mma":
	pair[boundary_matrix, vector[Finitely_critical_multi_filtration]] simplextree_to_boundary_filtration(uintptr_t)

def module_approximation_old(
		st:SimplexTreeMulti|None=None,
		max_error:float|None = None,
		box:list|np.ndarray|None = None,
		threshold:bool = False,
		complete:bool = True,
		multithread:bool = False, 
		verbose:bool = False,
		ignore_warning:bool = False,
		nlines:int = 500,
		max_dimension=np.inf,
		boundary = None,
		filtration = None,
		return_timings:bool = False,
		**kwargs
		):
	"""Computes an interval module approximation of a multiparameter filtration.

	Parameters
	----------
	st : n-filtered Simplextree, or None if boundary and filtration are provided.
		Defines the n-filtration on which to compute the homology.
	max_error: positive float
		Trade-off between approximation and computational complexity.
		Upper bound of the module approximation, in bottleneck distance, 
		for interval-decomposable modules.
	nlines: int
		Alternative to precision.
	box : pair of list of floats
		Defines a rectangle on which to compute the approximation.
		Format : [x,y], where x,y defines the rectangle {z : x ≤ z ≤ y}
	threshold: bool
		When true, intersects the module support with the box.
	verbose: bool
		Prints C++ infos.
	ignore_warning : bool
		Unless set to true, prevents computing on more than 10k lines. Useful to prevent a segmentation fault due to "infinite" recursion.
	return_timings : bool
		If true, will return the time to compute instead (computed in python, using perf_counter_ns).
	Returns
	-------
	PyModule
		An interval decomposable module approximation of the module defined by the
		homology of this multi-filtration.
	"""

	if boundary is None or filtration is None:
		from multipers.io import simplex_tree2boundary_filtrations
		boundary,filtration = simplex_tree2boundary_filtrations(st) # TODO : recomputed each time... maybe store this somewhere ?
	if max_dimension < np.inf: # TODO : make it more efficient
		nsplx = len(boundary)
		for i in range(nsplx-1,-1,-1):
			b = boundary[i]
			dim=len(b) -1
			if dim>max_dimension:
				boundary.pop(i)
				for f in filtration:
					f.pop(i)
	nfiltration = len(filtration)
	if nfiltration <= 0:
		return PyModule()
	if nfiltration == 1 and not(st is None):
		st = st.project_on_line(0)
		return st.persistence()

	if box is None and not(st is None):
		m,M = st.filtration_bounds()
	elif box is not None:
		m,M = np.asarray(box)
	else:
		m, M = np.min(filtration, axis=0), np.max(filtration, axis=0)
	prod = 1
	h = M[-1] - m[-1]
	for i, [a,b] in enumerate(zip(m,M)):
		if i == len(M)-1:	continue
		prod *= (b-a + h)

	if max_error is None:
		max_error:float = (prod/nlines)**(1/(nfiltration-1))

	if box is None:
		M = [np.max(f)+2*max_error for f in filtration]
		m = [np.min(f)-2*max_error for f in filtration]
		box = [m,M]

	if ignore_warning and prod >= 20_000:
		from warnings import warn
		warn(f"Warning : the number of lines (around {np.round(prod)}) may be too high. Try to increase the precision parameter, or set `ignore_warning=True` to compute this module. Returning the trivial module.")
		return PyModule()

	approx_mod = PyModule()
	cdef vector[Finitely_critical_multi_filtration] c_filtration = Finitely_critical_multi_filtration.from_python(filtration)
	cdef boundary_matrix c_boundary = boundary
	cdef value_type c_max_error = max_error
	cdef bool c_threshold = threshold
	cdef bool c_complete = complete
	cdef bool c_multithread = multithread
	cdef bool c_verbose = verbose
	cdef Box[value_type] c_box = Box[value_type](box)
	if return_timings:
		from time import perf_counter_ns
		t = perf_counter_ns()
	with nogil:
		c_mod = compute_vineyard_barcode_approximation(c_boundary,c_filtration,c_max_error, c_box, c_threshold, c_complete, c_multithread,c_verbose)
	if return_timings:
		t = perf_counter_ns() -t 
		t /= 10**9
		return t
	approx_mod._set_from_ptr(<intptr_t>(&c_mod))
	return approx_mod



def module_approximation(
		st:SimplexTreeMulti,
		value_type max_error = -1,
		box:list|np.ndarray|None = None,
		bool threshold:bool = False,
		bool complete:bool = True,
		bool verbose:bool = False,
		bool ignore_warning:bool = False,
		int nlines = 500,
		backend:str="matrix",
		# max_dimension=np.inf,
		# return_timings:bool = False,
		**kwargs
		):
	"""Computes an interval module approximation of a multiparameter filtration.

	Parameters
	----------
	st : n-filtered Simplextree, or None if boundary and filtration are provided.
		Defines the n-filtration on which to compute the homology.
	max_error: positive float
		Trade-off between approximation and computational complexity.
		Upper bound of the module approximation, in bottleneck distance, 
		for interval-decomposable modules.
	nlines: int
		Alternative to precision.
	box : pair of list of floats
		Defines a rectangle on which to compute the approximation.
		Format : [x,y], where x,y defines the rectangle {z : x ≤ z ≤ y}
	threshold: bool
		When true, intersects the module support with the box.
	verbose: bool
		Prints C++ infos.
	ignore_warning : bool
		Unless set to true, prevents computing on more than 10k lines. Useful to prevent a segmentation fault due to "infinite" recursion.
	return_timings : bool
		If true, will return the time to compute instead (computed in python, using perf_counter_ns).
	Returns
	-------
	PyModule
		An interval decomposable module approximation of the module defined by the
		homology of this multi-filtration.
	"""
	if backend == "old":
		if max_error == -1:
			max_error_=None
		else:
			max_error_=max_error
		return module_approximation_old(st, max_error=max_error_, box=box,threshold=threshold,complete=complete,verbose=verbose,ignore_warning=ignore_warning,nlines=nlines)



	cdef intptr_t ptr = st.thisptr
	cdef Simplex_tree_multi_interface* st_ptr = <Simplex_tree_multi_interface*>(ptr)
	cdef SimplicialVineGraphTruc graphtruc# copy ?
	cdef SimplicialVineMatrixTruc matrixtruc# copy ?
	cdef GeneralVineTruc generaltruc

	cdef int num_parameters = st.num_parameters

	if num_parameters <= 0:
		return PyModule()
	if num_parameters == 1 and not(st is None):
		st = st.project_on_line(0)
		return st.persistence()

	if box is not None:
		m,M = np.asarray(box)
	else:
		m,M = st.filtration_bounds()
	box =np.asarray([m,M])

	prod = 1
	h = M[-1] - m[-1]
	for i, [a,b] in enumerate(zip(m,M)):
		if i == len(M)-1:	continue
		prod *= (b-a + h)

	if max_error <= 0:
		max_error = (prod/nlines)**(1/(num_parameters-1))

	if not ignore_warning and prod >= 20_000:
		from warnings import warn
		warn(f"Warning : the number of lines (around {np.round(prod)}) may be too high. Try to increase the precision parameter, or set `ignore_warning=True` to compute this module. Returning the trivial module.")
		return PyModule()

	cdef Module mod
	cdef Box[value_type] c_box = Box[value_type](box)
	# Module multiparameter_module_approximation(Slicer &slicer, const value_type precision,
                                    # Box<value_type> &box, const bool threshold,
                                    # const bool complete, const bool verbose) 
	if backend == "matrix":
		matrixtruc = SimplicialVineMatrixTruc(st_ptr)
		with nogil:
			mod = multiparameter_module_approximation(matrixtruc, max_error,c_box,threshold, complete, verbose)
	elif backend == "graph":
		graphtruc = SimplicialVineGraphTruc(st_ptr)
		with nogil:
			mod = multiparameter_module_approximation(graphtruc, max_error,c_box,threshold, complete, verbose)
	else:
		raise ValueError("Invalid backend.")
	
	approx_mod = PyModule()
	approx_mod._set_from_ptr(<intptr_t>(&mod))
	return approx_mod



def multiparameter_module_approximation_from_slicer(slicer, box, int num_parameters, value_type max_error, bool complete, bool threshold, bool verbose):
	cdef intptr_t slicer_ptr = <intptr_t>(slicer.get_ptr())
	cdef GeneralVineTruc cslicer = dereference(<GeneralVineTruc*>(slicer_ptr))
	cdef Module mod
	cdef Box[value_type] c_box = Box[value_type](box)
	with nogil:
		mod = multiparameter_module_approximation(cslicer, max_error,c_box,threshold, complete, verbose)
	approx_mod = PyModule()
	approx_mod._set_from_ptr(<intptr_t>(&mod))
	return approx_mod
