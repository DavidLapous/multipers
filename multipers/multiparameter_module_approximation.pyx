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
from joblib import Parallel, delayed
import sys
from warnings import warn

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
from multipers.mma_structures import PyModule_type
from typing import Union
from multipers.slicer cimport _multiparameter_module_approximation_f32, _multiparameter_module_approximation_f64



def module_approximation_from_slicer(
        slicer:Slicer_type,
        box:Optional[np.ndarray]=None,
        max_error=-1,
        bool complete=True,
        bool threshold=False,
        bool verbose=False,
        list[float] direction = [],
        bool warnings = True,
        unsqueeze_grid = None,
        )->PyModule_type:

    cdef Module[float] mod_f32
    cdef Module[double] mod_f64
    cdef intptr_t ptr
    if not slicer.is_vine:
        if warnings:
            warn(r"(copy warning) Got a non-vine slicer as an input. Use `vineyard=True` to remove this copy.")
        from multipers._slicer_meta import Slicer
        slicer = Slicer(slicer, vineyard=True, backend="matrix")
    # if slicer.is_squeezed and unsqueeze_grid not None:
    #     raise ValueError("Got a squeezed slicer. Should have been unsqueezed before !")

    direction_ = np.asarray(direction, dtype=slicer.dtype)
    if slicer.dtype == np.float32:
        approx_mod = PyModule_f32()
        if box is None:
            box = slicer.filtration_bounds()
        mod_f32 = _multiparameter_module_approximation_f32(slicer,_py2p_f32(direction_), max_error,Box[float](box),threshold, complete, verbose)
        ptr = <intptr_t>(&mod_f32)
    elif slicer.dtype == np.float64:
        approx_mod = PyModule_f64()
        if box is None:
            box = slicer.filtration_bounds()
        mod_f64 = _multiparameter_module_approximation_f64(slicer,_py2p_f64(direction_), max_error,Box[double](box),threshold, complete, verbose)
        ptr = <intptr_t>(&mod_f64)
    else:
        raise ValueError(f"Slicer must be float-like. Got {slicer.dtype}.")

    approx_mod._set_from_ptr(ptr)

    if unsqueeze_grid is not None:
        if verbose:
            print("Reevaluating module in filtration grid...",end="", flush=True)
        approx_mod.evaluate_in_grid(unsqueeze_grid)
        from multipers.grids import compute_bounding_box
        if len(approx_mod):
            approx_mod.set_box(compute_bounding_box(approx_mod))
        if verbose:
            print("Done.",flush=True)

    return approx_mod

def module_approximation(
        input:Union[SimplexTreeMulti_type,Slicer_type, tuple],
        box:Optional[np.ndarray]=None,
        double max_error=-1, 
        int nlines=557,
        bool from_coordinates = False,
        bool complete=True, 
        bool threshold=False, 
        bool verbose=False,
        bool ignore_warnings=False,
        vector[double] direction = [],
        vector[int] swap_box_coords = [],
        *,
        int n_jobs = -1,
        )->PyModule_type:
    """Computes an interval module approximation of a multiparameter filtration.

    Parameters
    ----------
    input: SimplexTreeMulti or Slicer-like.
        Holds the multifiltered complex.
    max_error: positive float
        Trade-off between approximation and computational complexity.
        Upper bound of the module approximation, in bottleneck distance, 
        for interval-decomposable modules.
    nlines: int = 557
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
        
        assert all(is_slicer(s) and (s.is_minpres or len(s)==0) for s in input), "Modules cannot be merged unless they are minimal presentations."
        
        assert (np.unique([s.minpres_degree for s in input if len(s)], return_counts=True)[1] <=1).all(), "Multiple modules are at the same degree, cannot merge modules" 
        if len(input) == 0:
            return PyModule_f64()
        modules = tuple(Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(module_approximation)(
                input=slicer,
                box=box,
                max_error=max_error,
                nlines=nlines,
                from_coordinates=from_coordinates,
                complete=complete,
                threshold=threshold,
                verbose=verbose,
                ignore_warnings=ignore_warnings,
                direction=direction,
                swap_box_coords=swap_box_coords,
            )
            for slicer in input
        ))
        box = np.array([
            np.min([m.get_box()[0] for m in modules if len(m)], axis=0),
            np.max([m.get_box()[1] for m in modules if len(m)], axis=0),
        ])
        mod = PyModule_f64().set_box(box)
        for i,m in enumerate(modules):
            mod.merge(m, input[i].minpres_degree)
        return mod
    if len(input) == 0:
        if verbose:
            print("Empty input, returning the trivial module.")
        return PyModule_f64()


    cdef bool is_degenerate=False
    for i in range(direction.size()):
        if direction[i]<0:
            raise ValueError(f"Got an invalid negative direction. {direction=}")
        if direction[i] == 0:
            is_degenerate=True
    if is_degenerate and not ignore_warnings:
        warn("Got a degenerate direction. This function may fail if the first line is not generic.")

    if from_coordinates and not input.is_squeezed:
        input = input.grid_squeeze()
    unsqueeze_grid = None
    if input.is_squeezed:
        if not ignore_warnings:
            warn("(copy warning) Got a squeezed input. ")
        if verbose:
            print("Preparing filtration (unsqueeze)... ",end="", flush=True)
        if from_coordinates:
            from multipers.grids import sanitize_grid
            unsqueeze_grid = sanitize_grid(input.filtration_grid, numpyfy=True, add_inf=True)
            input = input.astype(dtype=np.float64)
            if direction.size() == 0:
                _direction = np.asarray([1/g.size for g in unsqueeze_grid], dtype=np.float64)
                _direction /= np.sqrt((_direction**2).sum())
                direction = _direction
            if verbose:
                print(f"Updated  `{direction=}`, and `{max_error=}` ",end="")

        else:
            input = input.unsqueeze()
        if verbose:
            print("Done.", flush=True)



    if box is None:
        if verbose:
            print("No box given. Using filtration bounds to infer it.")
        box = input.filtration_bounds()
        if verbose:
            print(f"Using {box=}.",flush=True)

    box = np.asarray(box)
    if box.ndim !=2:
        raise ValueError(f"Invalid box dimension. Got {box.ndim=} != 2")
    # empty coords
    zero_idx = box[1] == box[0]
    if np.any(zero_idx):
        if not ignore_warnings:
            warn(f"Got {(box[1] == box[0])=} trivial box coordinates.")
        box[1] += zero_idx

    for i in swap_box_coords:
        box[[0,1],i] = box[[1,0],i]
    num_parameters = box.shape[1]
    if num_parameters <=0:
        num_parameters = box.shape[1]
    assert direction.size() == 0 or direction.size() == box[0].size, f"Invalid line direction size, has to be 0 or {num_parameters=}"

    prod = sum(np.abs(box[1] - box[0])[:i].prod() * np.abs(box[1] - box[0])[i+1:].prod() for i in range(0,num_parameters) if (direction.size() ==0 or direction[i]!=0))

    if max_error <= 0:
        max_error = (prod/nlines)**(1/(num_parameters-1))

    estimated_nlines = prod/(max_error**(num_parameters -1))
    if not ignore_warnings and estimated_nlines >= 10_000:
        raise ValueError(f"""
Warning : the number of lines (around {np.round(estimated_nlines)}) may be too high. 
This may be due to extreme box or filtration bounds :

{box=}

Try to increase the precision parameter, or set `ignore_warnings=True` to compute this module. 
Returning the trivial module.
"""
        )
    if is_simplextree_multi(input):
        from multipers._slicer_meta import Slicer
        input = Slicer(input,backend="matrix", vineyard=True)
    if not is_slicer(input):
        raise ValueError("First argument must be a simplextree or a slicer !")
    return module_approximation_from_slicer(
            slicer=input,
            box=box,
            max_error=max_error,
            complete=complete,
            threshold=threshold,
            verbose=verbose,
            direction=direction,
            unsqueeze_grid=unsqueeze_grid,
            )

