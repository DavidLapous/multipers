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
from typing import Iterable,Optional, Literal
from cython.operator import dereference
#########################################################################
## Multipersistence Module Approximation Classes
from multipers.mma_structures cimport Module, Box, pair, boundary_matrix
from multipers.slicer cimport *


#########################################################################
## Small hack for typing
from gudhi import SimplexTree
from multipers.simplex_tree_multi import SimplexTreeMulti
from multipers.slicer import Slicer, SlicerClement,SlicerVineGraph,SlicerVineSimplicial
from multipers.mma_structures import PyModule
from typing import Union
import multipers
import multipers.io as mio
# cnp.import_array()

###################################### MMA
cdef extern from "multiparameter_module_approximation/approximation.h" namespace "Gudhi::multiparameter::mma":
    # TODO : tempita
    Module multiparameter_module_approximation(SimplicialVineGraphTruc&, value_type, Box[value_type]&, bool, bool, bool) nogil

    Module multiparameter_module_approximation(SimplicialVineMatrixTruc&, value_type, Box[value_type]&, bool, bool, bool) nogil
    Module multiparameter_module_approximation(GeneralVineTruc&, value_type, Box[value_type]&, bool, bool, bool) nogil
    Module multiparameter_module_approximation(GeneralVineClementTruc&, value_type, Box[value_type]&, bool, bool, bool) nogil








def module_approximation_from_slicer(
        slicer:Union[Slicer, SlicerClement,SlicerVineGraph,SlicerVineSimplicial], 
        box:Optional[np.ndarray]=None,
        value_type max_error=-1,
        bool complete=True,
        bool threshold=False,
        bool verbose=False,
        ):


    cdef intptr_t slicer_ptr = <intptr_t>(slicer.get_ptr())
    cdef GeneralVineTruc cslicer
    cdef GeneralVineClementTruc generalclementtruc
    cdef SimplicialVineGraphTruc graphtruc
    cdef SimplicialVineMatrixTruc matrixtruc
    cdef Module mod
    if box is None:
        box = slicer.compute_box()
    cdef Box[value_type] c_box = Box[value_type](box)
    if isinstance(slicer,Slicer):
        cslicer = dereference(<GeneralVineTruc*>(slicer_ptr))
        with nogil:
            mod = multiparameter_module_approximation(cslicer, max_error,c_box,threshold, complete, verbose)
    elif isinstance(slicer,SlicerClement):
        generalclementtruc = dereference(<GeneralVineClementTruc*>(slicer_ptr))
        with nogil:
            mod = multiparameter_module_approximation(generalclementtruc, max_error,c_box,threshold, complete, verbose)
    elif isinstance(slicer,SlicerVineGraph):
        graphtruc = dereference(<SimplicialVineGraphTruc*>(slicer_ptr))
        with nogil:
            mod = multiparameter_module_approximation(graphtruc, max_error,c_box,threshold, complete, verbose)
    elif isinstance(slicer,SlicerVineSimplicial):
        matrixtruc = dereference(<SimplicialVineMatrixTruc*>(slicer_ptr))
        with nogil:
            mod = multiparameter_module_approximation(matrixtruc, max_error,c_box,threshold, complete, verbose)
    else:
        raise ValueError("Unimplemeted slicer / Invalid slicer.")


    approx_mod = PyModule()
    approx_mod._set_from_ptr(<intptr_t>(&mod))
    return approx_mod

def module_approximation(
        input:Union[SimplexTreeMulti,Slicer, SlicerClement,SlicerVineGraph,SlicerVineSimplicial],
        box:Optional[np.ndarray]=None, 
        value_type max_error=-1, 
        int nlines=500,
        slicer_backend:Literal["matrix","clement","graph"]="matrix",
        minpres:Optional[Literal["mpfree"]]=None,
        degree:Optional[int]=None,
        bool complete=True, 
        bool threshold=False, 
        bool verbose=False,
        bool ignore_warning=False,
        id="",
        ):
    """Computes an interval module approximation of a multiparameter filtration.

    Parameters
    ----------
    input: SimplexTreeMulti or Slicer-like.
        Holds the multifiltered complex.
    box : (Optional) pair of list of floats
        Defines a rectangle on which to compute the approximation.
        Format : [x,y], where x,y defines the rectangle {z : x ≤ z ≤ y}
        If not given, takes the bounding box of the filtration.
    max_error: positive float
        Trade-off between approximation and computational complexity.
        Upper bound of the module approximation, in bottleneck distance, 
        for interval-decomposable modules.
    nlines: int = 200
        Alternative to precision;
        specifies the number of persistence computation used for the approximation.
    slicer_backend: Either "matrix","clement", or "graph".
        If a simplextree is given, it is first converted to this structure,
        with different choices of backends.
    minpres: (Optional) "mpfree" only for the moment.
        If given, and the input is a simplextree, 
        computes a minimal presentation before starting the computation.
    degree: int Only required when minpres is given.
        Homological degree of the minimal degree.
    threshold: bool
        When true, intersects the module support with the box,
        i.e. no more infinite summands.
    verbose: bool
        Prints C++ infos.
    ignore_warning : bool
        Unless set to true, prevents computing on more than 10k lines. 
        Useful to prevent a segmentation fault due to "infinite" recursion.
    Returns
    -------
    PyModule
        An interval decomposable module approximation of the module defined by the
        homology of this multi-filtration.
    """
    if box is None:
        if isinstance(input,SimplexTreeMulti):
            box = input.filtration_bounds()
        else:
            box = input.compute_box()
    box = np.asarray(box)
    num_parameters = box.shape[1]
    if num_parameters <=0:
        num_parameters = box.shape[1]

    m,M = box
    h = M[-1] - m[-1]
    prod = (M-m + h)[:-1].prod()
    if max_error <= 0:
        max_error = (prod/nlines)**(1/(num_parameters-1))

    if not ignore_warning and prod >= 10_000:
        raise ValueError(f"""
Warning : the number of lines (around {np.round(prod)}) may be too high. 
Try to increase the precision parameter, or set `ignore_warning=True` to compute this module. 
Returning the trivial module."""
        )
    if isinstance(input,SimplexTreeMulti):
        blocks = mio.simplextree2scc(input)
        if minpres is not None:
            mio.scc2disk(blocks, mio.input_path+id)
            blocks = mio.reduce_complex(mio.input_path+id, dimension=input.dimension-degree, backend=minpres)
        input = multipers.Slicer(input,backend=slicer_backend)

    return module_approximation_from_slicer(
            slicer=input,
            box=box,
            max_error=max_error,
            complete=complete,
            threshold=threshold,
            verbose=verbose,
            )





