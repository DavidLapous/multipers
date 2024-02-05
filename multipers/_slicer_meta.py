from typing import Literal, Optional


import multipers.slicer as mps
import multipers.simplex_tree_multi
from multipers.simplex_tree_multi import SimplexTreeMulti
import multipers.io as mio
import numpy as np


def _blocks2boundary_dimension_grades_box(blocks, filtration_type=np.float32):
    gen0_f = np.asarray(blocks[-3][0], dtype=filtration_type)
    gen1_f = np.asarray(blocks[-2][0], dtype=filtration_type)
    assert len(
        blocks[-1][0]) == 0, "Unimplemented when blocks[-1] is not trivial"
    gen0 = blocks[-3][1]
    gen1 = blocks[-2][1]
    multifiltration = np.concatenate([gen1_f, gen0_f])
    box = np.array(
        [
            [multifiltration[:, 0].min(), multifiltration[:, 1].min()],
            [multifiltration[:, 0].max(), multifiltration[:, 1].max()],
        ]
    )
    boundary = gen1 + gen0
    dimensions = np.array(([0] * len(gen1)) +
                          ([1] * len(gen0)), dtype=np.int32)
    return boundary, dimensions, multifiltration, box


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

    boundary, dimensions, multifiltrations, box = _blocks2boundary_dimension_grades_box(
        blocks
    )
    if vineyard:
        if backend == "matrix":
            return mps.Slicer(boundary, dimensions, multifiltrations)

        if backend == "clement":
            return mps.SlicerClement(boundary, dimensions, multifiltrations)
    if backend == "clement":
        raise ValueError("Clement is vineyard")

    raise ValueError("TODO : python interface for this")
