from copy import deepcopy
from typing import Literal, Optional

import numpy as np

import multipers.io as mio
import multipers.slicer as mps
from multipers.simplex_tree_multi import is_simplextree_multi


## TODO : maybe optimize this with cython
def _blocks2boundary_dimension_grades(
    blocks,
    filtration_type=np.float64,
    num_parameters: int = -1,
    inplace: bool = False,
    is_kcritical: bool = False,
):
    """
    Turns blocks, aka scc, into the input of non-simplicial slicers.
    """
    if num_parameters < 0:
        for b in blocks:
            if len(b[0]) > 0:
                if is_kcritical:
                    num_parameters = np.asarray(b[0][0]).shape[1]
                else:
                    num_parameters = np.asarray(b[0]).shape[1]
                break
        if num_parameters < 0:
            raise ValueError("Empty Filtration")
    rblocks = blocks if inplace else deepcopy(blocks)
    rblocks.reverse()
    block_sizes = [len(b[0]) for b in rblocks]
    S = np.cumsum([0, 0] + block_sizes)
    if is_kcritical:
        multifiltration = tuple(
            stuff
            for b in rblocks
            for stuff in (b[0] if len(b[0]) > 0 else [np.empty((0, num_parameters))])
        )

    else:
        multifiltration = np.concatenate(
            tuple(
                b[0] if len(b[0]) > 0 else np.empty((0, num_parameters))
                for b in rblocks
            ),
            dtype=filtration_type,
        )
    boundary = tuple(x + S[i] for i, b in enumerate(rblocks) for x in b[1])
    dimensions = np.fromiter(
        (i for i, b in enumerate(rblocks) for _ in range(len(b[0]))), dtype=int
    )
    return boundary, dimensions, multifiltration


def _slicer_from_simplextree(st, backend, vineyard):
    if vineyard:
        if backend == "matrix":
            slicer = mps._SlicerVineSimplicial(st)
        elif backend == "clement":
            raise ValueError("This one takes a minpres")
        elif backend == "graph":
            slicer = mps._SlicerVineGraph(st)
        else:
            raise ValueError(f"Inimplemented backend {backend}.")
    else:
        if backend == "matrix":
            slicer = mps._SlicerNoVineSimplicial(st)
        if backend == "clement":
            raise ValueError("Clement is Vineyard")
        if backend == "graph":
            raise ValueError("Graph is Vineyard")
    return slicer


def _slicer_from_blocks(
    blocks,
    backend,
    vineyard: bool,
    is_kcritical: bool,
    dtype: type,
    col: str,
):
    boundary, dimensions, multifiltrations = _blocks2boundary_dimension_grades(
        blocks,
        inplace=False,
        is_kcritical=is_kcritical,
    )
    if backend == "matrix":
        slicer = mps.get_matrix_slicer(vineyard, is_kcritical, dtype, col)(
            boundary, dimensions, multifiltrations
        )
    elif backend == "clement":
        assert dtype == np.float32 and not is_kcritical and vineyard
        slicer = mps._SlicerClement(boundary, dimensions, multifiltrations)
    else:
        raise ValueError(f"Inimplemented backend {backend}.")
    return slicer


def Slicer(
    st,
    backend: Literal["matrix", "clement", "graph"] = "matrix",
    vineyard: bool = True,
    reduce: bool = False,
    reduce_backend: Optional[str] = None,
    dtype=np.float64,
    is_kcritical: bool = False,
    column_type: str = "INTRUSIVE_SET",
) -> mps.Slicer_type:
    """
    Given a simplextree or blocks (a.k.a scc for python),
    returns a structure that can compute persistence on line (or more)
    slices, eventually vineyard update, etc.

    This can be used to compute interval-decomposable module approximations
    or signed measures, using, e.g.
     - `multipers.module_approximation(this, *args)`
     - `multipers.signed_measure(this, *args)`

    Note : it is recommended and sometime required to apply
        a minimal presentation before computing these functions !
    `mp.slicer.minimal_presentation(slicer, *args, **kwargs)`

    Input
    -----
     - st : SimplexTreeMulti or scc-like blocks or path to scc file
     - backend: slicer backend, e.g, "matrix", "clement", "graph"
     - vineyard: vineyard capable (may slow down computations if true)
    Output
    ------
    The corresponding slicer.
    """
    if mps.is_slicer(st):
        slicer = mps.get_matrix_slicer(vineyard, is_kcritical, dtype, column_type)(
            st.get_boundaries(), st.get_dimensions(), st.get_filtrations()
        )
        if st.is_squeezed:
            slicer.filtration_grid = st.filtration_grid
    elif is_simplextree_multi(st) and backend == "graph":
        slicer = _slicer_from_simplextree(st, backend, vineyard)
        if st.is_squeezed:
            slicer.filtration_grid = st.filtration_grid
    elif backend == "graph":
        raise ValueError(
            """
Graph is simplicial, incompatible with minpres.
You can try using `multipers.slicer.to_simplextree`."""
        )
    else:
        filtration_grid = None
        if is_simplextree_multi(st):
            blocks = st._to_scc()
            if st.is_squeezed:
                filtration_grid = st.filtration_grid
        elif isinstance(st, str):
            blocks = mio.scc_parser(st)
        else:
            blocks = st
        slicer = _slicer_from_blocks(
            blocks, backend, vineyard, is_kcritical, dtype, column_type
        )
        if filtration_grid is not None:
            slicer.filtration_grid = filtration_grid
    if reduce:
        slicer = mps.minimal_presentation(
            slicer, backend=reduce_backend, slicer_backend=backend, vineyard=vineyard
        )
    return slicer
