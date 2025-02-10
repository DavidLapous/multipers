from copy import deepcopy
from typing import  Optional

import numpy as np

import multipers.slicer as mps
from multipers.simplex_tree_multi import is_simplextree_multi
from multipers.slicer import _column_type, _valid_dtype, _valid_pers_backend, is_slicer


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
        elif backend == "clement":
            raise ValueError("Clement is Vineyard")
        elif backend == "graph":
            raise ValueError("Graph is Vineyard")
        else:
            raise ValueError(f"Inimplemented backend {backend}.")
    return slicer


def _slicer_from_blocks(
    blocks,
    pers_backend: _valid_pers_backend,
    vineyard: bool,
    is_kcritical: bool,
    dtype: type,
    col: _column_type,
):
    boundary, dimensions, multifiltrations = _blocks2boundary_dimension_grades(
        blocks,
        inplace=False,
        is_kcritical=is_kcritical,
    )
    slicer = mps.get_matrix_slicer(vineyard, is_kcritical, dtype, col, pers_backend)(
        boundary, dimensions, multifiltrations
    )
    return slicer


def Slicer(
    st=None,
    vineyard: Optional[bool] = None,
    reduce: bool = False,
    reduce_backend: Optional[str] = None,
    dtype: Optional[_valid_dtype] = None,
    kcritical: Optional[bool] = None,
    column_type: Optional[_column_type] = None,
    backend: Optional[_valid_pers_backend] = None,
    max_dim: Optional[int] = None,
    return_type_only: bool = False,
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

    if is_slicer(st, allow_minpres=False) or is_simplextree_multi(st):
        dtype = st.dtype if dtype is None else dtype
        is_kcritical = st.is_kcritical if kcritical is None else kcritical
    else:
        dtype = np.float64 if dtype is None else dtype
        is_kcritical = False if kcritical is None else kcritical

    if is_slicer(st, allow_minpres=False):
        vineyard = st.is_vine if vineyard is None else vineyard
        column_type = st.col_type if column_type is None else column_type
        backend = st.pers_backend if backend is None else backend
    else:
        vineyard = False if vineyard is None else vineyard
        column_type = "INTRUSIVE_SET" if column_type is None else column_type
        backend = "Matrix" if backend is None else backend

    _Slicer = mps.get_matrix_slicer(
        is_vineyard=vineyard,
        is_k_critical=is_kcritical,
        dtype=dtype,
        col=column_type,
        pers_backend=backend,
    )
    if return_type_only:
        return _Slicer
    if st is None:
        return _Slicer()
    elif mps.is_slicer(st):
        max_dim_idx = (
            None
            if max_dim is None
            else np.searchsorted(st.get_dimensions(), max_dim + 1)
        )
        slicer = _Slicer(
            st.get_boundaries()[slice(None, max_dim_idx)],
            st.get_dimensions()[slice(None, max_dim_idx)],
            st.get_filtrations()[slice(None, max_dim_idx)],
        )
        if st.is_squeezed:
            slicer.filtration_grid = st.filtration_grid
        slicer.minpres_degree = st.minpres_degree
    elif is_simplextree_multi(st) and backend == "Graph":
        slicer = _slicer_from_simplextree(st, backend, vineyard)
        if st.is_squeezed:
            slicer.filtration_grid = st.filtration_grid
    elif backend == "Graph":
        raise ValueError(
            """
Graph is simplicial, incompatible with minpres.
You can try using `multipers.slicer.to_simplextree`."""
        )
    else:
        filtration_grid = None
        if max_dim is not None:  # no test for simplex tree?
            st.prune_above_dimension(max_dim)
        if isinstance(st, str):  # is_kcritical should be false
            slicer = _Slicer()._build_from_scc_file(st)
        else:
            if is_simplextree_multi(st):
                blocks = st._to_scc()
                if st.is_squeezed:
                    filtration_grid = st.filtration_grid
            else:
                blocks = st
            slicer = _slicer_from_blocks(
                blocks, backend, vineyard, is_kcritical, dtype, column_type
            )
        if filtration_grid is not None:
            slicer.filtration_grid = filtration_grid
    if reduce:
        slicer = mps.minimal_presentation(
            slicer,
            backend=reduce_backend,
            slicer_backend=backend,
            vineyard=vineyard,
        )
    return slicer
