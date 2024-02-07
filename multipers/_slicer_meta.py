from typing import Literal, Optional


import multipers.slicer as mps
import multipers.simplex_tree_multi
from multipers.simplex_tree_multi import SimplexTreeMulti
import multipers.io as mio
import numpy as np
from copy import deepcopy


def _blocks2boundary_dimension_grades(
    blocks, filtration_type=np.float32, num_parameters: int = -1
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
    rblocks = deepcopy(blocks)
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


def Slicer(
    st: SimplexTreeMulti | list | str,
    backend: Literal["matrix", "clement", "graph"] = "matrix",
    vineyard: bool = True,
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

    st : SimplexTreeMulti or scc-like blocks
    backend: slicer backend;
    vineyard: vineyard capable (may slow down computations if true)
    """
    if isinstance(st, SimplexTreeMulti):
        if vineyard:
            if backend == "matrix":
                return mps.SlicerVineSimplicial(st)
            if backend == "clement":
                raise ValueError("This one takes a minpres")
            if backend == "graph":
                return mps.SlicerVineGraph(st)
            raise ValueError(f"Inimplemented backend {backend}.")
        if backend == "matrix":
            return mps.SlicerNoVineSimplicial(st)
        if backend == "clement":
            raise ValueError("Clement is Vineyard")
        if backend == "graph":
            raise ValueError("Graph is Vineyard")
    if backend == "graph":
        raise ValueError("Graph is simplicial, incompatible with minpres")
    if isinstance(st, SimplexTreeMulti):
        blocks = mio.minimal_presentation_from_mpfree(st)
    elif isinstance(st, str):
        blocks = mio.scc_parser(st)
    else:
        blocks = st

    boundary, dimensions, multifiltrations = _blocks2boundary_dimension_grades(blocks)
    if vineyard:
        if backend == "matrix":
            return mps.Slicer(boundary, dimensions, multifiltrations)

        if backend == "clement":
            return mps.SlicerClement(boundary, dimensions, multifiltrations)
    if backend == "matrix":
        return mps.SlicerNoVine(boundary, dimensions, multifiltrations)
    if backend == "clement":
        raise ValueError("Clement is vineyard")

    raise ValueError(f"Unimplemented combo : f{backend=}, f{vineyard=}")
