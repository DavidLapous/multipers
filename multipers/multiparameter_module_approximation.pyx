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
from multipers.mma_structures cimport *
from multipers.filtrations cimport *
from multipers.filtration_conversions cimport *
cimport numpy as cnp


#########################################################################
## Small hack for typing
from multipers.simplex_tree_multi import is_simplextree_multi, SimplexTreeMulti_type
from multipers.slicer import Slicer_type, is_slicer
from multipers.mma_structures import *
from typing import Union
import multipers
import multipers.io as mio
from multipers.slicer cimport _multiparameter_module_approximation_f32, _multiparameter_module_approximation_f64


def module_approximation_from_slicer(
        slicer:Slicer_type,
        box:Optional[np.ndarray]=None,
        max_error=-1,
        bool complete=True,
        bool threshold=False,
        bool verbose=False,
        list[float] direction = [],
        ):

    cdef Module[float] mod_f32
    cdef Module[double] mod_f64
    cdef intptr_t ptr
    if not slicer.is_vine:
        raise ValueError(f"Slicer must be able to do vineyards. Got {slicer}")
    if slicer.dtype == np.float32:
        approx_mod = PyModule_f32()
        if box is None:
            box = slicer.compute_box()
        mod_f32 = _multiparameter_module_approximation_f32(slicer,_py21c_f32(direction), max_error,Box[float](box),threshold, complete, verbose)
        ptr = <intptr_t>(&mod_f32)
    elif slicer.dtype == np.float64:
        approx_mod = PyModule_f64()
        if box is None:
            box = slicer.compute_box()
        mod_f64 = _multiparameter_module_approximation_f64(slicer,_py21c_f64(direction), max_error,Box[double](box),threshold, complete, verbose)
        ptr = <intptr_t>(&mod_f64)
    else:
        raise ValueError(f"Slicer must be float-like. Got {slicer.dtype}.")

    approx_mod._set_from_ptr(ptr)

    return approx_mod

def module_approximation(
        input:Union[SimplexTreeMulti_type,Slicer_type, tuple],
        box:Optional[np.ndarray]=None,
        float max_error=-1, 
        int nlines=500,
        slicer_backend:Literal["matrix","clement","graph"]="matrix",
        minpres:Optional[Literal["mpfree"]]=None,
        degree:Optional[int]=None,
        bool complete=True, 
        bool threshold=False, 
        bool verbose=False,
        bool ignore_warning=False,
        id="",
        list[float] direction = [],
        list[int] swap_box_coords = [],
        *,
        int n_jobs = 1,
        ):
    """Computes an interval module approximation of a multiparameter filtration.

    Parameters
    ----------
    input: SimplexTreeMulti or Slicer-like.
        Holds the multifiltered complex.
    max_error: positive float
        Trade-off between approximation and computational complexity.
        Upper bound of the module approximation, in bottleneck distance, 
        for interval-decomposable modules.
    nlines: int = 200
        Alternative to max_error;
        specifies the number of persistence computation used for the approximation.
    box : (Optional) pair of list of floats
        Defines a rectangle on which to compute the approximation.
        Format : [x,y], This defines a rectangle on which we draw the lines,
        uniformly drawn (with a max_error step).
        The first line is `x`. 
        **Warning**: For custom boxes, and directions, you **must** ensure
        that the first line captures a generic barcode.
    direction: float[:] = []
        If given, the line are drawn with this angle.
        **Warning**: You must ensure that the first line, defined by box,
        captures a generic barcode.
    slicer_backend: Either "matrix","clement", or "graph".
        If a simplextree is given, it is first converted to this structure,
        with different choices of backends.
    minpres: (Optional) "mpfree" only for the moment.
        If given, and the input is a simplextree, 
        computes a minimal presentation before starting the computation.
        A degree has to be given.
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
    if isinstance(input, tuple) or isinstance(input, list):
        if len(input) == 0:
            return PyModule_f64()
        if n_jobs <= 1: 
            modules = tuple(module_approximation(slicer, box, max_error, nlines, slicer_backend, minpres, degree, complete, threshold, verbose, ignore_warning, id, direction, swap_box_coords) for slicer in input)
        else:
            from joblib import Parallel, delayed
            modules = tuple(Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(module_approximation)(slicer, box, max_error, nlines, slicer_backend, minpres, degree, complete, threshold, verbose, ignore_warning, id, direction, swap_box_coords)
                for slicer in input
            ))
        mod = PyModule_f64().set_box(PyBox_f64(*modules[0].get_box()))
        for dim,m in enumerate(modules):
            mod.merge(m, dim)
        return mod
    if box is None:
        if is_simplextree_multi(input):
            box = input.filtration_bounds()
        else:
            box = input.compute_box()
    box = np.asarray(box)
    for i in swap_box_coords:
        box[0,i], box[1,i] = box[1,i], box[0,i]
    num_parameters = box.shape[1]
    if num_parameters <=0:
        num_parameters = box.shape[1]
    assert len(direction) == 0 or len(direction) == len(box[0]), f"Invalid line direction, has to be 0 or {num_parameters=}"

    prod = sum(np.abs(box[1] - box[0])[:i].prod() * np.abs(box[1] - box[0])[i+1:].prod() for i in range(0,num_parameters))

    if max_error <= 0:
        max_error = (prod/nlines)**(1/(num_parameters-1))

    if not ignore_warning and prod >= 10_000:
        raise ValueError(f"""
Warning : the number of lines (around {np.round(prod)}) may be too high. 
Try to increase the precision parameter, or set `ignore_warning=True` to compute this module. 
Returning the trivial module."""
        )
    if is_simplextree_multi(input):
        blocks = input._to_scc()
        if minpres is not None:
            assert not input.is_kcritical, "scc (and therefore mpfree, multi_chunk, 2pac, ...) format doesn't handle multi-critical filtrations."
            mio.scc2disk(blocks, mio.input_path+id)
            blocks = mio.reduce_complex(mio.input_path+id, dimension=input.dimension-degree, backend=minpres)
        else:
            pass
        input = multipers.Slicer(blocks,backend=slicer_backend, dtype = input.dtype, is_kcritical = input.is_kcritical())
    assert is_slicer(input), "First argument must be a simplextree or a slicer !"
    return module_approximation_from_slicer(
            slicer=input,
            box=box,
            max_error=max_error,
            complete=complete,
            threshold=threshold,
            verbose=verbose,
            direction=direction,
            )





