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
    minpres_backend: Optional[Literal["mpfree"]] = None,
    **minpres_backend_kwargs,
):
    if minpres_backend is None and isinstance(st, SimplexTreeMulti):
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
        if minpres_backend is None:
            blocks = mio.scc_parser(st)
        else:
            assert minpres_backend == "mpfree", "Mpfree only here"
            blocks = mio.minimal_presentation_from_str_mpfree(
                st, **minpres_backend_kwargs
            )
    else:
        blocks = st

    boundary, dimensions, multifiltrations = _blocks2boundary_dimension_grades(blocks)
    if vineyard:
        if backend == "matrix":
            return mps.Slicer(boundary, dimensions, multifiltrations)

        if backend == "clement":
            return mps.SlicerClement(boundary, dimensions, multifiltrations)
    if backend == "clement":
        raise ValueError("Clement is vineyard")

    raise ValueError("TODO : python interface for this")
