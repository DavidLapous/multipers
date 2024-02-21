from copy import deepcopy
from typing import Literal, Optional

import numpy as np

import multipers.io as mio
import multipers.slicer as mps
from multipers.simplex_tree_multi import SimplexTreeMulti


def _blocks2boundary_dimension_grades(
    blocks,
    filtration_type=np.float32,
    num_parameters: int = -1,
    inplace: bool = False,
):
    """
    Turns blocks, aka scc, into the input of non-simplicial slicers.
    """
    if num_parameters < 0:
        for b in blocks:
            if len(b[0]) > 0:
                num_parameters = np.asarray(b[0]).shape[1]
                break
        if num_parameters < 0:
            # empty presentation
            # return [], [], np.empty(0, dtype=filtration_type)
            raise ValueError("Empty Filtration")
    rblocks = blocks if inplace else deepcopy(blocks)
    rblocks.reverse()
    block_sizes = [len(b[0]) for b in rblocks]
    S = np.cumsum([0, 0] + block_sizes)
    multifiltration = np.concatenate(
        tuple(
            b[0] if len(b[0]) > 0 else np.empty((0, num_parameters)) for b in rblocks
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
            slicer = mps.SlicerVineSimplicial(st)
        elif backend == "clement":
            raise ValueError("This one takes a minpres")
        elif backend == "graph":
            slicer = mps.SlicerVineGraph(st)
        else:
            raise ValueError(f"Inimplemented backend {backend}.")
    else:
        if backend == "matrix":
            slicer = mps.SlicerNoVineSimplicial(st)
        if backend == "clement":
            raise ValueError("Clement is Vineyard")
        if backend == "graph":
            raise ValueError("Graph is Vineyard")
    return slicer


def _slicer_from_blocks(blocks, backend, vineyard):
    boundary, dimensions, multifiltrations = _blocks2boundary_dimension_grades(
        blocks,
        inplace=True,
    )
    if vineyard:
        if backend == "matrix":
            slicer = mps.Slicer(boundary, dimensions, multifiltrations)
        elif backend == "clement":
            slicer = mps.SlicerClement(boundary, dimensions, multifiltrations)
    else:
        if backend == "matrix":
            slicer = mps.SlicerNoVine(boundary, dimensions, multifiltrations)
        elif backend == "clement":
            raise ValueError("Clement is vineyard")
        raise ValueError(f"Unimplemented combo : f{backend=}, f{vineyard=}")
    return slicer


def Slicer(
    st: SimplexTreeMulti | list | str,
    backend: Literal["matrix", "clement", "graph"] = "matrix",
    vineyard: bool = True,
    reduce: bool = False,
    reduce_backend: Optional[str] = None,
):
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
    if isinstance(st, SimplexTreeMulti):
        slicer = _slicer_from_simplextree(st, backend, vineyard)
    elif backend == "graph":
        raise ValueError(
            """
Graph is simplicial, incompatible with minpres.
You can try using `multipers.slicer.to_simplextree`."""
        )
    else:
        if isinstance(st, str):
            blocks = mio.scc_parser(st)
        else:
            blocks = st
        slicer = _slicer_from_blocks(blocks, backend, vineyard)
    if reduce:
        slicer = mps.minimal_presentation(
            slicer, backend=reduce_backend, slicer_backend=backend, vineyard=vineyard
        )
    return slicer
